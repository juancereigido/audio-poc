import os, time, numpy as np
import sounddevice as sd
import soundfile   as sf
from pocketsphinx.pocketsphinx import Decoder
from pocketsphinx import get_model_path

# ––––– Load chime –––––
chime, fs_chime = sf.read("success.wav", dtype="int16")
chime = chime.reshape(-1)

# ––––– PocketSphinx keyword config –––––
model_path = get_model_path()
config = Decoder.default_config()
config.set_string('-hmm', os.path.join(model_path, 'en-us'))       # acoustic model
config.set_string('-keyphrase', 'porcupine')                        # your wake word
config.set_float ('-kws_threshold', 1e-20)                          # tweak sensitivity
decoder = Decoder(config)
decoder.start_utt()

# ––––– Stream state –––––
state        = "sleep"
chime_pos    = 0
start_time   = None
record_buf   = []

# ––––– Sample settings –––––
SAMPLE_RATE  = 16000
FRAME_LEN    = 512   # power-of-two, small enough for low latency

def callback(indata, outdata, frames, time_info, status):
    global state, chime_pos, start_time, record_buf

    pcm = indata[:,0].tobytes()
    mic  = indata[:,0].copy()
    out  = np.zeros(frames, dtype='int16')

    if state == "sleep":
        # feed chunks into Pocketsphinx
        decoder.process_raw(pcm, False, False)
        hyp = decoder.hyp()
        if hyp is not None:
            print("🔊 Wake word!")
            state     = "chime"
            chime_pos = 0
            decoder.end_utt()
    elif state == "chime":
        # play your sound.wav
        end   = chime_pos + frames
        chunk = chime[chime_pos:end]
        out[:len(chunk)] = chunk
        chime_pos += frames
        if chime_pos >= len(chime):
            state      = "record"
            start_time = time.time()
            record_buf = []
    elif state == "record":
        # capture 3 seconds
        record_buf.append(mic)
        if time.time() - start_time >= 3.0:
            # queue for playback
            callback.playback_buf = np.concatenate(record_buf)
            callback.playback_pos = 0
            state = "playback"
    elif state == "playback":
        # play back what you just recorded
        buf = callback.playback_buf
        pos = callback.playback_pos
        chunk = buf[pos:pos+frames]
        out[:len(chunk)] = chunk
        callback.playback_pos += frames
        if callback.playback_pos >= len(buf):
            # reset
            state = "sleep"
            decoder.start_utt()

    outdata[:] = out.reshape(-1,1)

# ––––– Open full-duplex stream ONCE –––––
stream = sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=FRAME_LEN,
    dtype='int16',
    channels=1,
    device=("hw:1,0","hw:1,0"),  # or your (in_dev, out_dev)
    callback=callback
)

try:
    stream.start()
    print("Say your wake word (‘porcupine’)…")
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    stream.stop()
    stream.close()
    decoder.end_utt()
    print("Goodbye.")
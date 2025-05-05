import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

from pocketsphinx import Decoder, Config
from pocketsphinx import get_model_path

# ─────── Configuration ───────
SAMPLE_RATE = 16000
FRAME_LEN   = 512
WAKE_WORD   = "success"
RECORD_SEC  = 3.0

# ─────── Load & Resample Chime ───────
y, sr_orig = sf.read("success.wav", always_2d=False)
if sr_orig != SAMPLE_RATE:
    y = resample_poly(y, SAMPLE_RATE, sr_orig, axis=0)
if y.ndim > 1:
    y = y.mean(axis=1)
chime = np.clip(y * 32767, -32768, 32767).astype("int16")

# ─────── PocketSphinx Setup ───────
model_path = get_model_path()
config     = Config()
config.set_string("-lm", None)
config.set_string("-hmm", os.path.join(model_path, "en-us", "en-us"))
config.set_string("-dict", os.path.join(model_path, "en-us", "cmudict-en-us.dict"))
config.set_string("-keyphrase", WAKE_WORD)
config.set_float ("-kws_threshold", 1e-20)
decoder = Decoder(config)
decoder.start_utt()

# ─────── Stream State ───────
state      = "sleep"
chime_pos  = 0
start_time = None
record_buf = []

def callback(indata, outdata, frames, time_info, status):
    global state, chime_pos, start_time, record_buf

    pcm = indata[:,0].copy()
    out = np.zeros(frames, dtype="int16")

    if state == "sleep":
        decoder.process_raw(pcm.tobytes(), False, False)
        if decoder.hyp() is not None:
            print("▶️  Wake word detected! Starting record+chime…")
            state      = "postwake"
            start_time = time.time()
            chime_pos  = 0
            record_buf = []
            decoder.end_utt()

    elif state == "postwake":
        # 1) record immediately
        record_buf.append(pcm)

        # 2) still play chime until it runs out
        end   = chime_pos + frames
        chunk = chime[chime_pos:end]
        out[:len(chunk)] = chunk
        chime_pos += frames

        # 3) when RECORD_SEC has passed, go to playback
        if time.time() - start_time >= RECORD_SEC:
            callback.playback_buf = np.concatenate(record_buf)
            callback.playback_pos = 0
            state = "playback"
            print(f"⏺️  {RECORD_SEC}s done — playing back…")

    elif state == "playback":
        buf = callback.playback_buf
        pos = callback.playback_pos
        chunk = buf[pos:pos+frames]
        out[:len(chunk)] = chunk
        callback.playback_pos += frames
        if callback.playback_pos >= len(buf):
            state = "sleep"
            decoder.start_utt()
            print("💤  Back to sleep")

    outdata[:] = out.reshape(-1,1)

# ─────── Open One Full-Duplex Stream ───────
stream = sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=FRAME_LEN,
    dtype="int16",
    channels=1,
    device=("hw:1,0","hw:1,0"),
    callback=callback
)

print(f"👂 Listening for “{WAKE_WORD}” (will record instantly)…")
stream.start()
try:
    while True:
        time.sleep(0.2)
except KeyboardInterrupt:
    pass
finally:
    stream.stop()
    stream.close()
    decoder.end_utt()
    print("🛑 Clean exit.")
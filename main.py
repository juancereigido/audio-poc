import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly        # <-- faster, no lazy loading

from pocketsphinx import Decoder, Config
from pocketsphinx import get_model_path

# ─────── Configuration ───────
SAMPLE_RATE = 16000
FRAME_LEN   = 512
WAKE_WORD   = "success"

# ─────── Load & Resample Chime ───────
y, sr_orig = sf.read("success.wav", always_2d=False)
print(f"Original file is {sr_orig} Hz, resampling to {SAMPLE_RATE} Hz…")

if sr_orig != SAMPLE_RATE:
    # integer-ratio resampling
    up   = SAMPLE_RATE
    down = sr_orig
    y = resample_poly(y, up, down, axis=0)

# force mono
if y.ndim > 1:
    y = y.mean(axis=1)

# convert to int16 PCM
chime = np.clip(y * 32767, -32768, 32767).astype("int16")
print(f"Chime loaded: {len(chime)} frames at {SAMPLE_RATE} Hz")

# ─────── PocketSphinx Setup ───────
model_path = get_model_path()
# 1) Start from a blank config (no LM/dict)
config = Config()
# Disable the default language model
config.set_string("-lm", None)
# 2) Point to the acoustic model (note the nested 'en-us/en-us' path)
config.set_string("-hmm", os.path.join(model_path, "en-us", "en-us"))
# 3) Point to the pronunciation dictionary
#    (needed even in keyphrase mode)
config.set_string("-dict", os.path.join(model_path, "en-us", "cmudict-en-us.dict"))
# 4) Turn on only the keyphrase spotter
config.set_string("-keyphrase", WAKE_WORD)
config.set_float ("-kws_threshold", 1e-20)
decoder = Decoder(config)
decoder.start_utt()

# ─────── Stream State ───────
state         = "sleep"
chime_pos     = 0
start_time    = None
record_buf    = []

def callback(indata, outdata, frames, time_info, status):
    global state, chime_pos, start_time, record_buf

    mic = indata[:, 0].copy()
    out = np.zeros(frames, dtype="int16")

    if state == "sleep":
        decoder.process_raw(mic.tobytes(), False, False)
        hyp = decoder.hyp()
        if hyp is not None:
            print("▶️  Wake word detected!")
            state     = "chime"
            chime_pos = 0
            decoder.end_utt()

    elif state == "chime":
        print("🔔  Playing chime…")
        end    = chime_pos + frames
        chunk  = chime[chime_pos:end]
        out[:len(chunk)] = chunk
        chime_pos += frames
        if chime_pos >= len(chime):
            state      = "record"
            start_time = time.time()
            record_buf = []
            print("⏺️  Recording for 3 seconds…")

    elif state == "record":
        record_buf.append(mic)
        if time.time() - start_time >= 3.0:
            buf = np.concatenate(record_buf)
            callback.playback_buf = buf
            callback.playback_pos = 0
            state = "playback"
            print("🔊  Playing back your recording…")

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

    outdata[:] = out.reshape(-1, 1)

# ─────── Open One Full-Duplex Stream ───────
stream = sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=FRAME_LEN,
    dtype="int16",
    channels=1,
    device=("hw:1,0", "hw:1,0"),  # your capture & playback device
    callback=callback,
)

try:
    stream.start()
    print(f"👂 Listening for “{WAKE_WORD}” on one full-duplex stream…")
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    stream.stop()
    stream.close()
    decoder.end_utt()
    print("🛑 Clean exit.")
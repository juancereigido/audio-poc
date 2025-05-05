import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

from pocketsphinx import Decoder
from pocketsphinx import get_model_path
import librosa   # <-- new dependency

# ─────── Configuration ───────
SAMPLE_RATE = 16000
FRAME_LEN   = 512       # small power-of-two for low latency
WAKE_WORD   = "porcupine"

# ─────── Load & Resample Chime ───────
# 1) Read original file and its samplerate
y, sr_orig = sf.read("success.wav", always_2d=False)

# 2) If needed, resample to SAMPLE_RATE
if sr_orig != SAMPLE_RATE:
    print(f"Resampling chime from {sr_orig} Hz to {SAMPLE_RATE} Hz…")
    y = librosa.resample(y.astype(float), orig_sr=sr_orig, target_sr=SAMPLE_RATE)

# 3) Ensure mono and convert to int16
if y.ndim > 1:
    y = y.mean(axis=1)
chime = np.clip(y * 32767, -32768, 32767).astype("int16")

# ─────── PocketSphinx Setup ───────
model_path = get_model_path()
config     = Decoder.default_config()
config.set_string("-hmm", os.path.join(model_path, "en-us"))
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
    pcm = mic.tobytes()

    if state == "sleep":
        decoder.process_raw(pcm, False, False)
        if decoder.hyp() is not None:
            print("🔊 Wake word detected!")
            state     = "chime"
            chime_pos = 0
            decoder.end_utt()

    elif state == "chime":
        end    = chime_pos + frames
        chunk  = chime[chime_pos:end]
        out[: len(chunk)] = chunk
        chime_pos += frames
        if chime_pos >= len(chime):
            state      = "record"
            start_time = time.time()
            record_buf = []

    elif state == "record":
        record_buf.append(mic)
        if time.time() - start_time >= 3.0:
            buf = np.concatenate(record_buf)
            callback.playback_buf = buf
            callback.playback_pos = 0
            state = "playback"

    elif state == "playback":
        buf = callback.playback_buf
        pos = callback.playback_pos
        chunk = buf[pos : pos + frames]
        out[: len(chunk)] = chunk
        callback.playback_pos += frames
        if callback.playback_pos >= len(buf):
            state = "sleep"
            decoder.start_utt()

    outdata[:] = out.reshape(-1, 1)

# ─────── Open One Full-Duplex Stream ───────
stream = sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=FRAME_LEN,
    dtype="int16",
    channels=1,
    device=("hw:1,0", "hw:1,0"),  # replace with your (in_dev, out_dev)
    callback=callback,
)

try:
    stream.start()
    print(f"Listening for “{WAKE_WORD}”…")
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    stream.stop()
    stream.close()
    decoder.end_utt()
    print("Clean exit.")
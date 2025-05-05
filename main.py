import pvporcupine, sounddevice as sd, soundfile as sf, numpy as np, time

# Load chime into a NumPy array
chime, fs_chime = sf.read("success.wav", dtype="int16")
chime = chime.reshape(-1)  # flatten in case it's (N,1)

# State machine
state = "sleep"
start_time = None
record_buf = []
chime_pos = 0

# Porcupine setup
access_key = "VG6zeSj7tpjIHVzyeWB7IfDqd9Qxfv5YXXIrlqnmp8rX5LVBbEBoxA=="  # Replace with your Picovoice access key
porcupine = pvporcupine.create(access_key=access_key, keywords=["picovoice"])
FRAME_LEN = porcupine.frame_length
SAMPLE_RATE = porcupine.sample_rate  # 16000

def callback(indata, outdata, frames, time_info, status):
    global state, start_time, record_buf, chime_pos

    pcm = indata[:,0].copy()
    out = np.zeros(frames, dtype='int16')

    if state == "sleep":
        # 1) Wake-word detection
        if porcupine.process(pcm.tobytes()) >= 0:
            state = "chime"
            chime_pos = 0
    elif state == "chime":
        # 2) Play chime
        end = chime_pos + frames
        chunk = chime[chime_pos:end]
        out[:len(chunk)] = chunk
        chime_pos += frames
        if chime_pos >= len(chime):
            # move to record phase
            state = "record"
            start_time = time.time()
            record_buf = []
    elif state == "record":
        # 3) Accumulate 3 seconds of mic audio
        record_buf.append(pcm)
        if time.time() - start_time >= 3.0:
            # switch to playback
            state = "playback"
            play_buf = np.concatenate(record_buf)
            record_buf = []  # free memory
            # store playback buffer and pos
            callback.playback_buf = play_buf
            callback.playback_pos = 0
    elif state == "playback":
        # 4) Output the just-recorded audio
        buf = callback.playback_buf
        pos = callback.playback_pos
        chunk = buf[pos:pos+frames]
        out[:len(chunk)] = chunk
        callback.playback_pos += frames
        if callback.playback_pos >= len(buf):
            # done, back to sleep
            state = "sleep"

    outdata[:] = out.reshape(-1,1)

# Open a single full-duplex stream
stream = sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=FRAME_LEN,
    dtype='int16',
    channels=1,
    device=("hw:1,0", "hw:1,0"),  # replace with your (card,device) tuple
    callback=callback
)

try:
    stream.start()
    print("Running! Say your wake word...")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    stream.stop()
    stream.close()
    porcupine.delete()
    print("Clean exit.")
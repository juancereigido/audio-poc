import pvporcupine
import sounddevice as sd
import soundfile as sf
import numpy as np

# 1) Load your chime once
chime, fs_chime = sf.read("success.wav", dtype="int16")

# 2) Configure Porcupine
porcupine = pvporcupine.create(keywords=["picovoice"])

def listen_for_wake():
    """Block until wake-word is heard."""
    buf_q = []
    with sd.InputStream(samplerate=porcupine.sample_rate,
                        blocksize=porcupine.frame_length,
                        channels=1, dtype="int16") as stream:
        print("→ Listening for wake word…")
        while True:
            data, _ = stream.read(porcupine.frame_length)
            if porcupine.process(data.tobytes()) >= 0:
                print("🔥 Wake word detected!")
                return

def play_chime():
    """Play the chime and block until done."""
    print("→ Playing chime…")
    sd.play(chime, fs_chime)
    sd.wait()

def record_and_playback(duration=3, fs=16000):
    """Record `duration` seconds, then play it back."""
    print(f"→ Recording for {duration} seconds…")
    recording = sd.rec(int(duration * fs), samplerate=fs,
                       channels=1, dtype="int16")
    sd.wait()
    print("→ Playing back your recording…")
    sd.play(recording, fs)
    sd.wait()

def main():
    try:
        while True:
            listen_for_wake()
            play_chime()
            record_and_playback()
    except KeyboardInterrupt:
        print("\nExiting…")
    finally:
        porcupine.delete()

if __name__ == "__main__":
    main()
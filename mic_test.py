import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 3

print("Recording 3 seconds...")
audio = sd.rec(int(DURATION * SAMPLE_RATE),
               samplerate=SAMPLE_RATE,
               channels=1,
               dtype="float32")
sd.wait()

print("Done!")
print("Max amplitude =", np.max(np.abs(audio)))
print("Mean amplitude =", np.mean(np.abs(audio)))

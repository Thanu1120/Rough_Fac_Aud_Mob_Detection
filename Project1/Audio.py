import pyaudio
import numpy as np

# Parameters
CHUNK = 1024  # Number of audio samples per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate in Hz
THRESHOLD = 500  # Volume threshold for warning

def detect_audio_and_warn():
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream for audio input
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening...")

    try:
        while True:
            # Read audio data from the stream
            data = stream.read(CHUNK)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Calculate the volume
            volume = np.linalg.norm(audio_data)

            # Check if volume exceeds the threshold
            if volume > THRESHOLD:
                print("Warning: High audio detected!")
    except KeyboardInterrupt:
        pass

    # Close the stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    detect_audio_and_warn()

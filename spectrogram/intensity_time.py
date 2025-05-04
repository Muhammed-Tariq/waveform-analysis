import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audioFile = r"/" # Insert file location in place of the '/'

frameSize = 2048
hopSize = 512

def plotSpectrogram(yPower, sr, hop, *, y_axis="log", size=(10, 4), dpi=100, title=None):
    yDB = librosa.power_to_db(yPower, ref=np.max, top_db=80)
    plt.figure(figsize=size, dpi=dpi)
    librosa.display.specshow(yDB, sr=sr, hop_length=hop, x_axis="time", y_axis=y_axis, cmap="magma")
    plt.colorbar(format="%+2.0fdB")
    if title:
        plt.title(title)
    plt.show()

if __name__ == "__main__":
    scale, sr = librosa.load(audioFile, sr=None)
    sScale = librosa.stft(scale, n_fft=frameSize, hop_length=hopSize)
    yScale = np.abs(sScale) ** 2
    plotSpectrogram(yScale, sr, hopSize, title="Intensity-Time Spectrogram")
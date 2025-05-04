import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audioFile = r"/" # Insert file location in place of the '/'

frameSize = 2048
hopSize = 512

def plotSpectrogram(y, sr, hopLength, yAxis="linear", title=None):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(y, sr=sr, hop_length=hopLength, x_axis="time", y_axis=yAxis)
    plt.colorbar(format="%+2.f")
    if title:
        plt.title(title)
    plt.show()

if __name__ == "__main__":
    scale, sr = librosa.load(audioFile)
    sScale = librosa.stft(scale, n_fft=frameSize, hop_length=hopSize)
    yScale = np.abs(sScale) ** 2
    yLogScale = librosa.power_to_db(yScale)
    plotSpectrogram(yLogScale, sr, hopSize, yAxis="log", title="Frequency-Time Spectrogram")
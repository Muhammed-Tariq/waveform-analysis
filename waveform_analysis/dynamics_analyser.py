from scipy.fftpack import fft
from scipy.io import wavfile
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Obtain length

def getLengthSeconds(path):
    y, sr = librosa.load(path, sr=None)
    return float(librosa.get_duration(y=y, sr=sr))

# Dynamics analysis

def amp(randomFiles, preferredFiles):
    randAvgAmp = []
    randAvgFluct = []
    randDynRange = []

    for i in range(len(randomFiles)):
        y, sr = librosa.load(randomFiles[i], sr=None)
        sAbsAvg = []
        for s in range(0, len(y), sr):
            sAbsAvg.append(np.abs(y[s:s + sr]).mean())
        randAvgAmp.append(float(np.mean(sAbsAvg)))
        randAvgFluct.append(float(np.std(sAbsAvg)))
        randDynRange.append(float(np.max(sAbsAvg) - np.min(sAbsAvg)))

    prefAvgAmp = []
    prefAvgFluct = []
    prefDynRange = []

    for j in range(len(preferredFiles)):
        y, sr = librosa.load(preferredFiles[j], sr=None)
        sAbsAvg = []
        for s in range(0, len(y), sr):
            sAbsAvg.append(np.abs(y[s:s + sr]).mean())
        prefAvgAmp.append(float(np.mean(sAbsAvg)))
        prefAvgFluct.append(float(np.std(sAbsAvg)))
        prefDynRange.append(float(np.max(sAbsAvg) - np.min(sAbsAvg)))
    return (randAvgAmp, 
        randAvgFluct, 
        randDynRange, 
        prefAvgAmp, 
        prefAvgFluct, 
        prefDynRange)

def main(randomFiles, preferredFiles):
    randLengths = [getLengthSeconds(p) for p in randomFiles]
    prefLengths = [getLengthSeconds(p) for p in preferredFiles]
    (randAvgAmp, 
    randAvgFluct, 
    randDynRange, 
    prefAvgAmp, 
    prefAvgFluct, 
    prefDynRange) = amp(randomFiles, preferredFiles)
    return (randLengths,
    randAvgAmp, 
    randAvgFluct, 
    randDynRange,
    prefLengths,
    prefAvgAmp, 
    prefAvgFluct, 
    prefDynRange)

if __name__ == "__main__":
    main()
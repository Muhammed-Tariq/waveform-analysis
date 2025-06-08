from waveform_analysis.upload import randomFiles, preferredFiles
import numpy as np
import librosa
import librosa.display

randAvgAmp = []
randAvgFluct = []
randDynRange = []

for i in range(len(randomFiles)):
    y, sr = librosa.load(randomFiles[i])
    sAbsAvg = []
    for s in range(0, len(y), sr):
        sAbsAvg.append(np.abs(y[s:s + sr]).mean())
    randAvgAmp.append(np.mean(sAbsAvg))
    randAvgFluct.append(np.std(sAbsAvg))
    randDynRange.append(np.max(sAbsAvg) - np.min(sAbsAvg))

prefAvgAmp = []
prefAvgFluct = []
prefDynRange = []

for j in range(len(preferredFiles)):
    y, sr = librosa.load(preferredFiles[j])
    sAbsAvg = []
    for s in range(0, len(y), sr):
        sAbsAvg.append(np.abs(y[s:s + sr]).mean())
    prefAvgAmp.append(np.mean(sAbsAvg))
    prefAvgFluct.append(np.std(sAbsAvg))
    prefDynRange.append(np.max(sAbsAvg) - np.min(sAbsAvg))

print(prefAvgAmp)
print(prefAvgFluct)
print(prefDynRange)

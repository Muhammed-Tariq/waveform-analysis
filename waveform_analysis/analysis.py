from pathlib import Path
import numpy as np
import librosa
import librosa.display

# Upload audio files

def getAudioFiles(root, extension):
    files = [p for p in root.rglob("*") if p.suffix.lower() in extension]
    print(f"Found {len(files)} audio files.")
    return files

randomRoot = Path("S:\Downloads\Music\Random")
preferredRoot = Path("S:\Downloads\Music\Preferred")
extension = (".wav",)

randomFiles = getAudioFiles(randomRoot, extension)
preferredFiles = getAudioFiles(preferredRoot, extension)

# Audio analysis

# Length

def getLengthSeconds(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return float(librosa.get_duration(y=y, sr=sr))

randLengths = [getLengthSeconds(p) for p in randomFiles]
prefLengths = [getLengthSeconds(p) for p in preferredFiles]

# Average amplitude, average fluctuation, dynamic range

def amp(randomFiles, preferredFiles):
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
    return randAvgAmp, randAvgFluct, randDynRange, prefAvgAmp, prefAvgFluct, prefDynRange

# Optional for debugging

randAvgAmp, randAvgFluct, randDynRange, prefAvgAmp, prefAvgFluct, prefDynRange = amp(randomFiles, preferredFiles)

print("Random song lengths:", randLengths)
print("Random song average amplitudes:", randAvgAmp)
print("Random song average fluctuations:", randAvgFluct)
print("Random song dynamic ranges:", randDynRange)

print("Preferred song lengths:", prefLengths)
print("Preferred song average amplitudes:", prefAvgAmp)
print("Preferred song average fluctuations:", prefAvgFluct)
print("Preferred song dynamic ranges:", prefDynRange)
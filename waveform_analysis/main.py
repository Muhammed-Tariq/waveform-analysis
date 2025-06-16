from waveform_analysis import consonance_analyser as ca
from waveform_analysis import dynamics_analyser as da
from pathlib import Path

# Obtain audio files

def getAudioFiles(root, extension):
    files = [p for p in root.rglob("*") if p.suffix.lower() in extension]
    print(f"Found {len(files)} audio files.")
    return files

randomRoot = Path("S:\Downloads\Music\Random")
preferredRoot = Path("S:\Downloads\Music\Preferred")
extension = (".wav",)

randomFiles = getAudioFiles(randomRoot, extension)
preferredFiles = getAudioFiles(preferredRoot, extension)

# Obtain consonance statistics

randomConsonanceList = []

for i in range(len(randomFiles)):
    randomConsonanceList.append(ca.main(randomFiles[i]))
print(randomConsonanceList)
from pathlib import Path

def getAudioFiles(root, extension):
    files = [p for p in root.rglob("*") if p.suffix.lower() in extension]
    print(f"Found {len(files)} audio files.")
    return files

randomRoot = Path("S:\Downloads\Music\Random")
preferredRoot = Path("S:\Downloads\Music\Preferred")
extension = (".wav")

randomFiles = getAudioFiles(randomRoot, extension)
preferredFiles = getAudioFiles(preferredRoot, extension)
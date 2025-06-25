#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# Tunable constants
FPS = 4                     # Visualiser FPS (default 4 = 0.25s per frame)
TOP_NOTES = 3               # Maximum notes displayed on the visualisation
WINDOW_SEC = 0.25           # Scrolling‑window length (seconds)
MIN_HZ_SEPARATION = 16      # Ignore peaks closer than this (Hz)
FREQ_MIN, FREQ_MAX = 10, 2048
MIN_WEIGHT = 0.20           # Ignore peaks <20 % of frame max when finding ratios
MIN_REL_AMP = 0.05          # ignore FFT bins <5 % of frame max  (≈ −26 dB)
HARM_TOL_CENTS = 6          # Treat ±N ¢ as the same harmonic
MAX_HARMONIC_NUM = 6        # Search up to n‑th harmonic for flagging
CENTS_TOL = 6               # Max error (¢) allowed for ratio simplification
MAX_DEN_SEARCH = 32         # How far we search for denominator when simplifying
MAX_PRINT = 15              # Pairs kept for optional printing

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
]
LOG2 = np.log2

# Utility helpers
def freqToNote(freq: float) -> str:
    """Return closest equal‑temperament note name (e.g. 'A4')."""
    if freq <= 0:
        return "—"
    semitones = 69 + 12 * LOG2(freq / 440.0)
    idx = int(round(semitones))
    name = NOTE_NAMES[idx % 12]
    octave = idx // 12 - 1  # fix: use integer division, not float
    return f"{name}{octave}"


def cents(ratio: float) -> float:
    """Convert a frequency ratio to cents."""
    return 1200 * LOG2(ratio)

# Main program
def main(wavFile: str | Path) -> float:
    """Analyse *wavFile* and return its weighted consonance score."""
    wavFile = Path(wavFile)

    fs, data = wavfile.read(wavFile)
    audio = data if data.ndim == 1 else data[:, 0]  # Keep left channel if stereo

    winLen = int(fs * WINDOW_SEC)
    hannWin = np.hanning(winLen) # Pre‑computed window
    freqs = np.fft.rfftfreq(winLen, 1 / fs)

    totalFrames = int(len(audio) / fs * FPS)
    samplesPerFrame = int(fs / FPS)

    # Prepare matplotlib figure
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    line, = ax.plot(freqs, np.zeros_like(freqs))
    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.set_ylim(0, 1.05)
    title = ax.set_title("")
    noteTexts = [
        ax.text(0, 0.9 - k * 0.07, "", ha="center", va="bottom", fontsize=14)
        for k in range(TOP_NOTES)
    ]

    # Create a folder for the (optional) visualisation frames
    outdir = wavFile.with_suffix("").with_name(f"{wavFile.stem}_frames")
    outdir.mkdir(parents=True, exist_ok=True)

    # Global consonance accumulators
    totalWeightedSum = 0.0
    totalWeight = 0.0

    # Frame loop
    print(f"Rendering {totalFrames} frames …")
    for frameIdx in range(totalFrames):
        start = frameIdx * samplesPerFrame
        seg = audio[start:start + winLen]
        if len(seg) < winLen:
            seg = np.pad(seg, (0, winLen - len(seg)))

        # FFT and magnitude spectrum for this frame
        mag = np.abs(np.fft.rfft(seg * hannWin))
        if not mag.any():
            continue
        maxAmp = mag.max()
        if maxAmp == 0:
            continue

        # Detect local‑max peaks that are loud enough
        rawPeaks: list[tuple[float, str, float]] = []  # (freq, note, weight)
        for idx in range(1, len(mag) - 1):
            amp = mag[idx]
            if amp < MIN_REL_AMP * maxAmp:
                continue
            if amp < mag[idx - 1] or amp < mag[idx + 1]:
                continue
            freq = freqs[idx]
            rawPeaks.append((freq, freqToNote(freq), amp / maxAmp))

        noteList = [(f, n, w) for f, n, w in rawPeaks if w >= MIN_WEIGHT]
        if len(noteList) < 2:  # Need at least one interval
            continue

        # Interval processing & consonance metrics
        for a in range(len(noteList)):
            f1, n1, w1 = noteList[a]
            for b in range(a + 1, len(noteList)):
                f2, n2, w2 = noteList[b]
                if n1 == n2:
                    continue

                if f1 >= f2:
                    fHi, wHi, fLo, wLo = f1, w1, f2, w2
                else:
                    fHi, wHi, fLo, wLo = f2, w2, f1, w1

                rawRatio = fHi / fLo

                # Try to approximate ratio to a simple fraction within CENTS_TOL.
                bestFrac: Fraction | None = None
                for d in range(1, MAX_DEN_SEARCH + 1):
                    n = round(rawRatio * d)
                    if n == 0:
                        continue
                    frac = Fraction(n, d)
                    approxVal = frac.numerator / frac.denominator
                    if abs(cents(rawRatio / approxVal)) <= CENTS_TOL:
                        bestFrac = frac
                        break
                if bestFrac is None:
                    continue

                relevance = np.sqrt(wHi * wLo)  # Geometric mean of weights
                complexity = bestFrac.numerator + bestFrac.denominator
                totalWeightedSum += relevance * complexity
                totalWeight += relevance

        # Optional visualisation
        # Find strongest *note* peaks for labeling, ensuring separation
        peakCandidates = sorted(enumerate(mag), key=lambda x: x[1], reverse=True)
        peaks: list[tuple[float, str]] = []
        for idx, _amp in peakCandidates:
            freq = freqs[idx]
            if any(abs(freq - p[0]) < MIN_HZ_SEPARATION for p in peaks):
                continue
            peaks.append((freq, freqToNote(freq)))
            if len(peaks) == TOP_NOTES:
                break

        # Update re‑usable figure
        line.set_ydata(mag / maxAmp)
        currentTime = frameIdx / FPS
        mainLabel = peaks[0][1] if peaks else "—"
        title.set_text(
            f"{wavFile.name} • t={currentTime:.1f}s • note: {mainLabel}")

        for k, txt in enumerate(noteTexts):
            if k < len(peaks):
                f, lbl = peaks[k]
                txt.set_text(lbl)
                txt.set_position((f, 0.9 - k * 0.07))
            else:
                txt.set_text("")

        # Uncomment to dump PNG frames:
        # fig.savefig(outdir / f"frame_{frameIdx:04d}.png")

    plt.close(fig)

    # Final weighted consonance score
    if not totalWeight:
        print("No intervals passed the filters; no consonance score computed.")
        return float("nan")

    # Lower complexity = more consonant. Inverted so that higher = more consonant.
    avgWeight = totalWeightedSum / totalWeight
    consonanceScore = 10 * (1.0 / avgWeight)

    print(
        "Weighted consonance score for", Path(wavFile).stem, "(1 / avg(numerator+denominator), weight‑weighted): "
        f"{consonanceScore:.4f}")
    return float(consonanceScore)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute a weighted consonance score for a .wav file.")
    parser.add_argument("wavFile", type=Path, help="Path to input .wav file")
    args = parser.parse_args()
    main(args.wavFile)

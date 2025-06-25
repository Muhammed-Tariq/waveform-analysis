#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Tunable constants (behaviour kept identical to the original script)
# ---------------------------------------------------------------------------
FPS = 4                     # visualiser FPS (default 4 ≃ 0.25 s per frame)
TOP_NOTES = 3               # maximum notes displayed on the visualisation
WINDOW_SEC = 0.25           # scrolling‑window length (seconds)
MIN_HZ_SEPARATION = 16      # suppress peaks closer than this (Hz)
FREQ_MIN, FREQ_MAX = 10, 2048
MIN_WEIGHT = 0.20           # ignore peaks <20 % of frame max when finding ratios
MIN_REL_AMP = 0.05          # ignore FFT bins <5 % of frame max  (≈ −26 dB)
HARM_TOL_CENTS = 6          # treat ±6 ¢ as the same harmonic
MAX_HARMONIC_NUM = 6        # search up to n‑th harmonic for flagging
CENTS_TOL = 6               # max tuning error (¢) allowed for ratio simplification
MAX_DEN_SEARCH = 32         # how far we search for denominator when simplifying
MAX_PRINT = 15              # pairs kept for optional printing

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
]
LOG2 = np.log2  # tiny micro‑optimisation

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def freq_to_note(freq: float) -> str:
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

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main(wav_file: str | Path) -> float:
    """Analyse *wav_file* and return its weighted consonance score."""
    wav_file = Path(wav_file)

    # ---------------------------------------------------------------------
    # I/O & pre‑allocation
    # ---------------------------------------------------------------------
    fs, data = wavfile.read(wav_file)
    audio = data if data.ndim == 1 else data[:, 0]  # keep left channel if stereo

    win_len = int(fs * WINDOW_SEC)
    hann_win = np.hanning(win_len)                  # pre‑computed window
    freqs = np.fft.rfftfreq(win_len, 1 / fs)

    total_frames = int(len(audio) / fs * FPS)
    samples_per_frame = int(fs / FPS)

    # Prepare matplotlib figure (unchanged)
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    line, = ax.plot(freqs, np.zeros_like(freqs))
    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.set_ylim(0, 1.05)
    title = ax.set_title("")
    note_texts = [
        ax.text(0, 0.9 - k * 0.07, "", ha="center", va="bottom", fontsize=14)
        for k in range(TOP_NOTES)
    ]

    # Create a sibling folder for the (optional) visualisation frames
    outdir = wav_file.with_suffix("").with_name(f"{wav_file.stem}_frames")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Global consonance accumulators
    # ---------------------------------------------------------------------
    total_weighted_complexity = 0.0
    total_weight = 0.0

    # ---------------------------------------------------------------------
    # Frame loop
    # ---------------------------------------------------------------------
    print(f"Rendering {total_frames} frames …")
    for frame_idx in range(total_frames):
        start = frame_idx * samples_per_frame
        seg = audio[start:start + win_len]
        if len(seg) < win_len:
            seg = np.pad(seg, (0, win_len - len(seg)))

        # FFT & magnitude spectrum for this frame
        mag = np.abs(np.fft.rfft(seg * hann_win))
        if not mag.any():
            continue
        max_amp = mag.max()
        if max_amp == 0:
            continue

        # -----------------------------------------------------------------
        # Detect local‑max peaks that are loud enough
        # -----------------------------------------------------------------
        raw_peaks: list[tuple[float, str, float]] = []  # (freq, note, weight)
        for idx in range(1, len(mag) - 1):
            amp = mag[idx]
            if amp < MIN_REL_AMP * max_amp:
                continue
            if amp < mag[idx - 1] or amp < mag[idx + 1]:
                continue
            freq = freqs[idx]
            raw_peaks.append((freq, freq_to_note(freq), amp / max_amp))

        note_list = [(f, n, w) for f, n, w in raw_peaks if w >= MIN_WEIGHT]
        if len(note_list) < 2:  # need at least one interval
            continue

        # -----------------------------------------------------------------
        # Interval processing & consonance metrics
        # -----------------------------------------------------------------
        for a in range(len(note_list)):
            f1, n1, w1 = note_list[a]
            for b in range(a + 1, len(note_list)):
                f2, n2, w2 = note_list[b]
                if n1 == n2:
                    continue

                # high / low assignment (simplifies later maths)
                if f1 >= f2:
                    f_hi, w_hi, f_lo, w_lo = f1, w1, f2, w2
                else:
                    f_hi, w_hi, f_lo, w_lo = f2, w2, f1, w1

                raw_ratio = f_hi / f_lo

                # Try to approximate ratio with a simple fraction within CENTS_TOL.
                best_frac: Fraction | None = None
                for d in range(1, MAX_DEN_SEARCH + 1):
                    n = round(raw_ratio * d)
                    if n == 0:
                        continue
                    frac = Fraction(n, d)
                    approx_val = frac.numerator / frac.denominator
                    if abs(cents(raw_ratio / approx_val)) <= CENTS_TOL:
                        best_frac = frac
                        break
                if best_frac is None:
                    continue

                relevance = np.sqrt(w_hi * w_lo)  # geometric mean of weights
                complexity = best_frac.numerator + best_frac.denominator
                total_weighted_complexity += relevance * complexity
                total_weight += relevance

        # -----------------------------------------------------------------
        # Optional visualisation (kept intact, mostly unchanged)
        # -----------------------------------------------------------------
        # find strongest *note* peaks for labeling, ensuring separation
        peak_candidates = sorted(enumerate(mag), key=lambda x: x[1], reverse=True)
        peaks: list[tuple[float, str]] = []
        for idx, _amp in peak_candidates:
            freq = freqs[idx]
            if any(abs(freq - p[0]) < MIN_HZ_SEPARATION for p in peaks):
                continue
            peaks.append((freq, freq_to_note(freq)))
            if len(peaks) == TOP_NOTES:
                break

        # Update re‑usable figure
        line.set_ydata(mag / max_amp)
        current_time = frame_idx / FPS
        main_label = peaks[0][1] if peaks else "—"
        title.set_text(
            f"{wav_file.name} • t={current_time:.1f}s • note: {main_label}")

        for k, txt in enumerate(note_texts):
            if k < len(peaks):
                f, lbl = peaks[k]
                txt.set_text(lbl)
                txt.set_position((f, 0.9 - k * 0.07))
            else:
                txt.set_text("")

        # Uncomment to dump PNG frames:
        # fig.savefig(outdir / f"frame_{frame_idx:04d}.png")

    plt.close(fig)

    # ---------------------------------------------------------------------
    # Final weighted consonance score
    # ---------------------------------------------------------------------
    if not total_weight:
        print("No intervals passed the filters; no consonance score computed.")
        return float("nan")

    # Lower complexity ⇒ more consonant. Invert so that higher = more consonant.
    avg_complexity = total_weighted_complexity / total_weight
    consonance_score = 10 * (1.0 / avg_complexity)

    print(
        "Weighted consonance score for", Path(wav_file).stem, "(1 / avg(numerator+denominator), weight‑weighted): "
        f"{consonance_score:.4f}")
    return float(consonance_score)


# ---------------------------------------------------------------------------
# CLI entry‑point (keeps import‑ability of ``main``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute a weighted consonance score for a .wav file.")
    parser.add_argument("wav_file", type=Path, help="Path to input .wav file")
    args = parser.parse_args()
    main(args.wav_file)

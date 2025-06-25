from waveform_analysis import consonance_analyser as ca
from waveform_analysis import dynamics_analyser as da
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
import math

# Constants

DISSONANCE_WEIGHTING = 2.0
LENGTH_WEIGHTING = 0.6

AMP_WEIGHT = 1
FLUCT_WEIGHT = 1
DYN_RANGE_WEIGHT = 1.5

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

randConsonanceList = []
preferredConsonanceList = []

for i in range(len(randomFiles)):
    randConsonanceList.append(ca.main(randomFiles[i]))

for i in range(len(preferredFiles)):
    preferredConsonanceList.append(ca.main(preferredFiles[i]))

# Obtain dynamics statistics

(randLengths,
randAvgAmp, 
randAvgFluct, 
randDynRange,
prefLengths,
prefAvgAmp, 
prefAvgFluct, 
prefDynRange) = da.main(randomFiles, preferredFiles)

# Obtain scores

def squash(x: Decimal) -> Decimal:
    s = x.sqrt()
    return s / (s + 1)

def calculateSimilarity(cons, length, amp, fluct, dynRange):
    with localcontext() as ctx:
        ctx.prec = 25

        consScore   = squash(Decimal(cons))
        lengthScore = squash(Decimal(length))
        dynRangeScore = squash(Decimal(dynRange))
        ampScore    = squash(Decimal(amp))
        fluctScore  = squash(Decimal(fluct))

        ampScore    = float(ampScore) * AMP_WEIGHT
        fluctScore  = float(fluctScore) * FLUCT_WEIGHT
        dynRangeScore = float(dynRangeScore) * DYN_RANGE_WEIGHT

        lengthScore = float(lengthScore) * LENGTH_WEIGHTING
        consScore   = float(consScore) * DISSONANCE_WEIGHTING

    return float(ampScore + 
                 fluctScore + 
                 dynRangeScore + 
                 lengthScore + 
                 consScore)

prefSimilarityScores = [calculateSimilarity(cons, length, amp, fluct, dyn)
    for cons, length, amp, fluct, dyn in zip(
        preferredConsonanceList,
        prefLengths,
        prefAvgAmp,
        prefAvgFluct,
        prefDynRange
    )
]

avgPreferredSimilarity = sum(prefSimilarityScores)/len(prefSimilarityScores)

randSimilarityScores = [calculateSimilarity(cons, length, amp, fluct, dyn)
for cons, length, amp, fluct, dyn in zip(
        randConsonanceList,
        randLengths,
        randAvgAmp,
        randAvgFluct,
        randDynRange
    )
]

ranked_pairs = sorted(
    (
        (fname, abs(score - avgPreferredSimilarity))
        for fname, score in zip(randomFiles, randSimilarityScores)
    ),
    key=lambda t: t[1]
)

print("\nRandom songs ranked (closest to farthest):")
for pos, (fname, gap) in enumerate(ranked_pairs, 1):
    print(f"{pos:>2}. {fname.name:<50}  gap={gap:.4f}")


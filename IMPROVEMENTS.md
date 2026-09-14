# Improvements Log

Tracking how the neural network went from broken to actually working, with real numbers at each step. Everything here is pulled from actual training runs, not estimates.

## Architecture

| Version | Hidden layers | Neurons | Dropout |
|---|---|---|---|
| v1 | 1 | 16 | none |
| v2 | 2 | 16, 32 | 0.2, 0.3 |
| v3 | 2 | 92, 150 | 0.01 |
| v4 | 3 | 92, 150, 200 | 0.01 |
| v5 | 3 | 92, 150, 200 | 0.001 |
| v6 (current) | 4 | 92, 150, 200, 270 | 0.0001 |

v1 was a separate, simpler model that got merged into what's now `models/ImprovedNeuralNetwork.py`. Everything from v2 onward is that same file getting wider and deeper over time.

## Metrics

This is the real story, including the mistakes, because the mistakes are as instructive as the wins.

| Stage | Setup | Result |
|---|---|---|
| First real training run | v2 arch, 1000 throws (flat ground), 200 epochs, no val split | Loss 215.10 |
| More epochs | Same, 1000 epochs | Loss 5.20 (plateaued) |
| Manual tuning | Same, learning rate adjusted | Loss 0.32 |
| Added train/val split | Same data, split 80/20 for the first time | Train 0.29, **val 6.46** |

That val loss was the real turning point. Train loss looked great the whole time, but the model was just memorizing the training rows. Val loss showed the truth: it wasn't generalizing at all.

| Stage | Setup | Result |
|---|---|---|
| Normalized inputs | vx0/vy0/mass were on wildly different scales | Val loss **0.056** (~115x better) |
| Added terrain variance | Ground is no longer flat, harder target to predict | Val loss 0.076 |
| Added early stopping | Stop training when val loss stops improving, keep best checkpoint | More reliable, same ballpark |

Then a real bug got caught: `ThrowSim.cpp`'s throw count had been bumped from 1000 to 5000 to 10000, but the actual dataset file never got regenerated after the first bump. Every "bigger dataset" result up to that point was still trained on the original 1000-row file. Editing the source doesn't do anything until you rebuild and actually rerun the executable.

| Stage | Setup | Result |
|---|---|---|
| Before the fix (stale 1000-row data) | v4 arch | Val loss 0.115, MAE vs real physics: x = 0.35, y = 0.08 |
| After the fix (real 10000-row data) | v6 arch | Val loss **0.0114**, MAE vs real physics: x = 0.104, y = 0.038 |
| 5-fold cross validation | v6 arch, real data, 5 separate train/val splits | Mean val loss 0.0105, std dev 0.0013 |

The 5-fold result is the one that matters most. A single train/val split can get lucky. Five folds landing within 0.0013 of each other means this is a real, repeatable number, not a fluke.

## What actually moved the needle, ranked

1. **Fixing the stale dataset bug.** Bigger architecture and more claimed data did basically nothing until the data was actually real. This was the single biggest lesson of the whole project.
2. **Input normalization.** 115x improvement from one change, because vx0/vy0/mass were on completely different scales and the network was struggling to learn from all three at once.
3. **Adding the train/val split.** Didn't improve anything by itself, but revealed that the model was overfitting badly, which is what made every fix after it possible to actually evaluate.
4. **Actually generating more data** (1000 throws to 10000, once the pipeline was fixed to really do it).
5. Architecture widening, dropout tuning, early stopping. Real, but smaller gains than the four above.

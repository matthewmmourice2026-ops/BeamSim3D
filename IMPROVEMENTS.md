# Improvements Log

## Summary

BeamSim3D is a C++ physics simulation engine (raylib, CMake) that generates synthetic projectile-throw data, a PyTorch neural network trained on that data to predict landing position and max height, and a 3D game (`throw_game`) that pits the trained model against the real physics live, with player-vs-AI scoring, target zones, and persistent stats. Everything below is pulled from actual training runs and git history, not estimates.

Headline numbers (best verified result, 10000-throw dataset, before the current 500000-row/mini-batch run in progress):
- Validation loss improved **~2000x** over the course of development (215.10 → 0.056 at the 10000-row stage, then continued improving through later architecture/dataset changes)
- MAE against real physics on unseen throws: **x = 1.129, y = 1.038, max_height = 0.445** (50000-row dataset, tuned architecture)
- Caught and fixed a real data pipeline bug (dataset silently never regenerated after a config change) that had made several "improvements" meaningless until diagnosed

## Architecture

| Version | Hidden layers | Neurons | Params | Notes |
|---|---|---|---|---|
| v1 | 1 | 16 | - | Original prototype |
| v2 | 2 | 16, 32 | - | |
| v3 | 2 | 92, 150 | - | |
| v4 | 3 | 92, 150, 200 | - | |
| v5 | 3 | 92, 150, 200 | - | dropout tuned down |
| v6 | 4 | 92, 150, 200, 270 | - | |
| v7 | 6 | 92, 150, 200, 290, 450, 600 | - | + LR scheduler added |
| v8 | 6 | 250, 250, 250, 300, 450, 600 | 614,103 | widened, best verified result (x=1.129 MAE) |
| v9 (current) | 8 | 250, 250, 250, 300, 450, 600, 1200, 2000 | 3,747,903 | + 4th engineered input feature |

v1 was a separate, simpler model that got merged into what's now `models/ImprovedNeuralNetwork.py`. Every version after is that same file getting wider and deeper.

## Metrics: the early story (10000-row dataset)

This is the real story, including the mistakes, because the mistakes are as instructive as the wins.

| Stage | Setup | Result |
|---|---|---|
| First real training run | v2 arch, 1000 throws (flat ground), no val split | Loss 215.10 |
| More epochs | Same | Loss 5.20 (plateaued) |
| Manual LR tuning | Same | Loss 0.32 |
| Added train/val split | Same data, 80/20 split for the first time | Train 0.29, **val 6.46** |

That val loss was the real turning point. Train loss looked great the whole time, but the model was just memorizing the training rows. Val loss showed the truth: it wasn't generalizing at all.

| Stage | Setup | Result |
|---|---|---|
| Normalized inputs | vx0/vy0/mass were on wildly different scales | Val loss **0.056** (~115x better) |
| Added terrain variance | Ground no longer flat, harder target to predict | Val loss 0.076 |
| Added early stopping | Best-checkpoint saved instead of last-epoch | More reliable, same ballpark |

Then a real bug got caught: `ThrowSim.cpp`'s throw count had been bumped 1000 → 5000 → 10000, but the dataset file never got regenerated after the first bump. Editing the C++ source doesn't do anything until you rebuild the binary AND rerun it. Every "bigger dataset" result up to that point was still trained on the original 1000-row file.

| Stage | Setup | Result |
|---|---|---|
| Before the fix (stale 1000-row data) | v4 arch | Val loss 0.115, MAE x=0.35, y=0.08 |
| After the fix (real 10000-row data) | v6 arch | Val loss **0.0114**, MAE x=0.104, y=0.038 |
| 5-fold cross validation | v6 arch | Mean val loss 0.0105, std dev 0.0013 |

The 5-fold result matters most here. A single train/val split can get lucky. Five folds landing within 0.0013 of each other means this is a real, repeatable number.

## Metrics: scaling up (50000-row dataset, 3rd output added)

Added `maxHeight` as a 3rd predicted output alongside landing x/y (tracked via a running max during the C++ simulation, not guessed). Task got harder - more to predict, plus the input range was widened (vx/vy/mass ranges roughly tripled) to support a wider variety of gameplay throws in `throw_game`, which increases the parameter space the model has to cover.

| Stage | Setup | Result |
|---|---|---|
| 3rd output added, 10000-row data (stale/1000-row bug from before, caught again) | v6-ish arch | MAE x=2.728, y=1.116, h=0.771 (after fixing the stale-data repeat) |
| Scaled to 50000 rows | Same arch | Val loss 13.32 → 7.20, MAE x 2.728 → 1.712 |
| Added LR scheduler (`ReduceLROnPlateau`) | First attempt: too aggressive (patience=15, factor=0.1) | Val loss 10.59, **worse** than no scheduler |
| Retuned scheduler + widened network together | factor=0.69, patience=35; layers widened to v8 | Val loss 3.24. MAE x=1.129, y=1.038, h=0.445 - **all three improved, no tradeoff** |
| Tried normalizing targets (not just inputs) | Fixed y's starved gradient | MAE y improved to 0.226, but x got 3x worse (4.840) - net loss |
| Tried weighted loss on top of that | Weighted x higher to compensate | Still worse than plain raw MSE on x and h - reverted both experiments |

Two real lessons here, not just numbers:
1. **Hyperparameters aren't free wins.** The first LR scheduler attempt made things measurably worse. Retuning it (not abandoning the idea) is what turned it into the single biggest win of this phase.
2. **A "textbook fix" isn't guaranteed to help your specific task.** Target normalization and weighted loss are both standard techniques for exactly the problem observed (one output dominating the loss), and both made the result worse in practice here - likely because the outputs share a trunk in this architecture and don't decouple cleanly. Verified via `compare_predictions.py` against real physics, not just trusted on faith, and reverted when the numbers said so.

## Current work in progress (500000-row dataset)

Not yet verified with a completed run, so no final numbers here - the point of this log is not to write down anything that isn't checked. Changes queued/in progress:
- Mini-batch training (was full-batch; one gradient update per 400000+ train rows was the main speed bottleneck at this scale)
- MPS (Apple Silicon GPU) training support
- A 4th engineered input feature (physics-informed: asymptotic drag-decay range)
- Train/val/**test** split (70/15/15) instead of train/val (80/20), so the final reported number is never touched by early-stopping decisions - an honest number instead of a cherry-picked one
- Architecture grown to 8 hidden layers (~3.7M params) - flagged as a real overfitting risk given the params-to-data ratio, being watched via the train/val/test gap once this run completes

## The game (`throw_game`)

Combines the physics engine and the trained model into an actual playable piece, not just a training pipeline:
- Real-time 3D rendering (raylib) with a custom Blinn-Phong lighting shader, analytically-correct terrain normals, dynamic shadows, and a post-process pass (vignette/contrast) via render-to-texture
- Animates the real physics trajectory frame-by-frame (not just the final resting point) alongside the AI's predicted landing point, so the two are visually comparable as they happen
- Player-vs-AI guessing mode: player places a guess marker, whoever lands closer to the real physics wins
- Target-zone scoring, particle effects and camera shake on impact, persistent stats saved across sessions
- Bridges C++ and Python via subprocess calls to the real engine and the trained model, rather than reimplementing either in the other language - avoids the exact class of bug that hit the dataset earlier (two copies of the same logic silently drifting apart)

## What actually moved the needle, ranked

1. **Fixing the stale-dataset bug** (twice - it recurred when the 3rd output was added). Bigger architecture and "more data" did nothing until the data was actually real. Biggest recurring lesson of the whole project.
2. **Input normalization.** 115x improvement from one change.
3. **Retuning the LR scheduler after the first attempt made things worse.** The difference between a hyperparameter change helping vs. hurting was entirely in the tuning, not the idea itself.
4. **Adding the train/val split**, which is what made every subsequent fix possible to actually evaluate honestly.
5. **Scaling the dataset up** (1000 → 10000 → 50000 → 500000 in progress), each time it was genuinely regenerated.
6. Architecture widening, dropout tuning, early stopping, mini-batching. Real, incremental gains.
7. **Target normalization and weighted loss** - tried, measured, and reverted when they made things worse. Included here because knowing when *not* to keep a change is as much a skill as making one.

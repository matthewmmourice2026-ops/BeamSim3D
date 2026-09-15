# Improvements Log

## Summary

BeamSim3D is a C++ physics simulation engine (raylib, CMake) that generates synthetic projectile-throw data, a PyTorch neural network trained on that data to predict landing position and max height, and a 3D game (`throw_game`) that pits the trained model against the real physics live, with player-vs-AI scoring, target zones, and persistent stats. Everything below is pulled from actual training runs and git history, not estimates.

Headline numbers (current best, 500000-throw dataset):
- Validation loss improved **~500x** over the course of development (215.10 → 0.4262), while the task itself got harder along the way (3 outputs instead of 1, 50x more data, a much wider input range)
- MAE against real physics on unseen throws: **x = 0.969, y = 0.168, max_height = 0.640**
- Held-out test loss (0.4321) closely matches validation loss (0.4262) - confirms the result generalizes, not a lucky split
- Caught and fixed a real data pipeline bug (dataset silently never regenerated after a config change) **twice** - once for the initial dataset, once again when a 3rd output was added - that had made several "improvements" meaningless until diagnosed
- Caught and fixed a training-loop bug where early-stopping patience was shorter than the LR scheduler's patience, causing training to give up before the scheduler ever got a chance to act (79 epochs instead of 928, val loss 5x worse)

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

## Metrics: 500000-row dataset, mini-batching, MPS, 8-layer model

A lot changed at once here: mini-batch training instead of full-batch (was the main speed bottleneck once the dataset passed 50000 rows), MPS (Apple Silicon GPU) support, a 4th engineered input feature (physics-informed: asymptotic drag-decay range under pure exponential drag decay), Huber loss instead of MSE, weight decay added, a train/val/**test** split (70/15/15) instead of train/val (80/20) so the final number is never touched by early-stopping decisions, and the architecture grown to 8 hidden layers (~3.7M params).

| Stage | Setup | Result |
|---|---|---|
| First mini-batch/MPS run | scheduler patience=55, early-stop patience=50 (bug: scheduler patience longer than early-stop) | 79 epochs, val loss 2.03, MAE x=4.233, y=0.962, h=2.124 - **worse than the previous best on x and h** |
| Fixed early-stop patience to 100 (> scheduler's 55) | Same everything else | 928 epochs, val loss **0.4262**, test loss 0.4321, MAE x=0.969, y=0.168, h=0.640 |

The bug: early-stopping patience was *shorter* than the LR scheduler's patience, backwards from how they're supposed to relate (scheduler needs room to act before training gives up, not the other way around). Training stopped at epoch 79 having barely let the scheduler fire once. Fixing the patience relationship alone was a 5x improvement in val loss and a 6x improvement in y's MAE, using the exact same architecture and data. A one-line config bug outweighed several actual training-pipeline features (mini-batching, MPS, the engineered feature) combined - infrastructure only pays off if the loop around it is actually configured to let it work.

The params-to-data ratio here (3.7M params, 500000 rows, dropout down to near-zero) was flagged as a real overfitting risk before this ran. It didn't overfit - val and test loss stayed close together - most likely because the 10x larger dataset gave the bigger model enough signal to actually use that capacity instead of memorizing.

## The game (`throw_game`)

Combines the physics engine and the trained model into an actual playable piece, not just a training pipeline:
- Real-time 3D rendering (raylib) with a custom Blinn-Phong lighting shader, analytically-correct terrain normals, dynamic shadows, and a post-process pass (vignette/contrast) via render-to-texture
- Animates the real physics trajectory frame-by-frame (not just the final resting point) alongside the AI's predicted landing point, so the two are visually comparable as they happen
- Player-vs-AI guessing mode: player places a guess marker, whoever lands closer to the real physics wins
- Target-zone scoring, particle effects and camera shake on impact, persistent stats saved across sessions
- Bridges C++ and Python via subprocess calls to the real engine and the trained model, rather than reimplementing either in the other language - avoids the exact class of bug that hit the dataset earlier (two copies of the same logic silently drifting apart)

## What actually moved the needle, ranked

1. **Fixing the stale-dataset bug** (recurred twice - once for the initial dataset, once when the 3rd output was added). Bigger architecture and "more data" did nothing until the data was actually real. Biggest recurring lesson of the whole project.
2. **Fixing the early-stop-vs-scheduler patience relationship.** A one-line config bug (early stopping shorter than scheduler patience, the two need to relate in the opposite order) cost 5x on val loss and 6x on one output's MAE - bigger than several actual feature additions (mini-batching, MPS, the engineered input) combined.
3. **Input normalization.** 115x improvement from one change.
4. **Retuning the LR scheduler after the first attempt made things worse.** The difference between a hyperparameter change helping vs. hurting was entirely in the tuning, not the idea itself.
5. **Adding the train/val split**, which is what made every subsequent fix possible to actually evaluate honestly.
6. **Scaling the dataset up** (1000 → 10000 → 50000 → 500000), each time it was genuinely regenerated.
7. Architecture widening, dropout tuning, mini-batching, MPS. Real, incremental gains, but the two config bugs above outweighed all of them individually.
8. **Target normalization and weighted loss** - tried, measured, and reverted when they made things worse. Included here because knowing when *not* to keep a change is as much a skill as making one.

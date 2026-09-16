# BeamSim3D

A C++ physics engine that generates synthetic projectile-throw data, a PyTorch neural network trained to predict where things land (plus its own confidence), and a 3D game that puts the trained model up against the real physics live, with scoring and a player-vs-AI mode.

## What's in here

Three parts, in order of how they build on each other.

**The C++ physics engine** (`ThrowSim.cpp`) simulates an object thrown with a given velocity, mass, launch height, and wind, integrated frame by frame with gravity, drag, and a bounce off rolling-hill terrain. It has a batch mode that fires hundreds of thousands of randomized throws and logs `vx0, vy0, mass, height0, windAccel -> final_x, final_y, maxHeight, timeToLand, bounceCount, apexTime, finalVx`, and a single-throw mode for asking the real physics for ground truth on one specific input.

**The ML pipeline** (`main.py` and friends) trains a neural network on that dataset to predict landing position and flight characteristics from the launch parameters, plus a self-estimate of how confident it is — then checks the result against the real engine rather than trusting the training loss number.

**The game** (`ThrowGame.cpp`) renders both the real throw and the AI's predicted landing live in a 3D window (raylib, custom Blinn-Phong lighting shaders, distance fog, particle effects), lets you play against the AI (guess the landing spot, closest wins), and tracks target-zone scoring and stats across sessions. Adjustable camera (mouse orbit/zoom), pause, and a settings panel are built in.

There's also a soft-body vehicle physics demo (`main.cpp`, `Physics.cpp`) that predates the throw/ML side of the project - a separate raylib window (`beam_sim`) showing a mass-spring vehicle chassis.

## Building the C++ part

Needs CMake and raylib. On macOS with Homebrew:

```
brew install raylib cmake
```

Then from the project root:

```
mkdir -p build
cd build
cmake ..
make
```

Builds three executables:

- `beam_sim` - the vehicle physics demo
- `throw_sim` - headless data generator, also has single-throw and trajectory ground-truth modes
- `throw_game` - the actual game

Run `./throw_sim` first to generate a dataset, then `./throw_game` to play (needs a trained model - see below - or it'll just show the real physics with the AI prediction marked as unavailable).

`throw_sim` usage:
```
./throw_sim                                                          # batch mode, writes throw_results.csv
./throw_sim <vx0> <vy0> <mass> <height0> <windAccel>                 # single throw, prints all 7 outputs
./throw_sim --trajectory <vx0> <vy0> <mass> <height0> <windAccel>    # prints every simulation step (animates the throw in-game)
```

One thing that bit me more than once: editing `throwCount` (or anything else) in `ThrowSim.cpp` does nothing on its own. You have to rebuild (`make throw_sim`) AND actually rerun it (`./throw_sim`) or `throw_results.csv` stays exactly as it was. Lost real time chasing "why isn't more data helping" before realizing I was retraining on a stale file more than once.

## Playing the game

```
cd build
./throw_game
```

Controls:

```
LEFT/RIGHT/UP/DOWN   adjust throw velocity
[ / ]                adjust mass
A / D                move your landing guess (blue disc)
SPACE                throw
R                     replay the last throw
F11                   fullscreen
ESC                   pause
TAB                   settings (mouse sensitivity, invert Y)
right-drag / scroll   orbit / zoom the camera
```

A magenta flag marks a random target zone each throw - land close to it for points. Your guess vs. the AI's prediction, whoever's closer to the real landing spot wins that round. Stats persist across sessions in `game_stats.txt`, camera/mouse settings in `game_settings.txt` (both at the repo root).

Needs your Python venv active (see below) for `predict_server.py` to start - without it, the game still runs and shows real physics, just with the AI side marked unavailable instead of crashing. Every real throw you play also gets logged to `game_throws.csv` at the repo root (same schema as `throw_results.csv`) - `main.py`/`kfold_eval.py` pick it up automatically and train on it alongside the synthetic dataset if it exists, so playing the game actually grows the training set over time. Nothing trains *during* play, though - the model stays frozen for inference; you have to rerun `main.py` for played throws to actually affect the checkpoint.

## The machine learning part

Needs pandas and PyTorch:

```
python3 -m venv venv
source venv/bin/activate
pip install pandas torch
```

Train:

```
python3 main.py
```

Loads `build/throw_results.csv` (plus `game_throws.csv` if present), splits it 70/15/15 into train/val/test, normalizes inputs (train-set stats only, no leakage), trains with mini-batches, a learning-rate scheduler, and early stopping. Val loss (combined with an uncertainty-calibration term) drives early stopping; test loss is reported separately at the end since it's never touched during training - an honest number instead of one cherry-picked by early stopping. Saves weights + normalization stats to `models/ImprovedNeuralNetwork.pth`. Uses Apple Silicon's MPS backend automatically if available, falls back to CPU otherwise.

Check the model against real physics (not just the training loss number):

```
python3 compare_predictions.py
```

Runs 30 fixed unseen throws through both `throw_sim` and the trained model, prints them side by side, reports mean absolute error per output, and checks whether the model's self-reported uncertainty actually tracks its real landing-position error.

More solid accuracy number than one train/val split:

```
python3 kfold_eval.py
```

5-fold cross validation, mirrors `main.py`'s training setup so the numbers are comparable. Takes a while - it's training 5 separate models.

Quick single prediction from the command line:

```
python3 predict.py <vx0> <vy0> <mass> <height0> <windAccel>
```

Prints all 8 outputs (7 regression targets + uncertainty) as comma-separated floats. This is a cold-start, one-shot version - `throw_game` instead talks to `predict_server.py`, a persistent version of the same thing kept alive for the whole game session so it only pays the torch-import/checkpoint-load cost once instead of once per throw.

## How the physics data is generated

Each throw starts at a given launch height with a random horizontal velocity, vertical velocity, mass, and horizontal wind acceleration, integrated with gravity and linear drag. It lands on rolling-hill terrain (two sine waves added together) instead of flat ground, bounces with some energy loss and ground friction, and settles once it's basically stopped moving (or the simulation cap is hit - strong wind can prevent full settling, which is real behavior of the simplified model, not a bug). `final_x`, `final_y`, `maxHeight`, `timeToLand`, `bounceCount`, `apexTime`, and `finalVx` all get logged as the targets the network has to predict.

## The model

Feedforward network (`models/ImprovedNeuralNetwork.py`), 9 hidden layers (250→250→250→300→450→600→1200→2000→2000), ReLU + BatchNorm + 0.05 dropout after each, **7,764,408 parameters**. Takes 6 inputs: the 5 raw launch parameters plus one physics-informed engineered feature computed in `features.py` (shared by every script that touches the model so they can't drift out of sync) - `x_inf`, the asymptotic drag-decay range. Outputs 8 values: `final_x, final_y, maxHeight, timeToLand, bounceCount, apexTime, finalVx`, plus a log-variance uncertainty head trained via a separate Gaussian NLL loss so the model reports a calibrated confidence alongside each prediction.

Current best result, trained on 1,000,000 throws:

- Best val loss 0.9621, test loss 0.9647 (held-out set never touched during training - close agreement to val means this generalizes, not a lucky split)
- `final_x` MAE 4.258, uncertainty calibration correlation 0.757
- `final_x` is the weakest output relative to the others and the subject of a still-open investigation (`IMPROVEMENTS.md` section 8) - a best-ever run once reached val loss 0.2968 / `final_x` MAE 1.871, but that checkpoint was lost to a training run overwriting it before being backed up, and hasn't been fully recovered yet. An added `physics_baseline_x` engineered feature (still defined in `features.py`, currently unused) was tried and reverted - it regressed every output on the run it was tested with, confounded with an unrelated patience change made the same session; worth retrying in isolation.

`IMPROVEMENTS.md` has the full history of what actually moved these numbers, including the things that didn't work and got reverted, and a checkpoint-safety lesson worth reading before touching `main.py` again.

## Repo layout

```
main.cpp / Physics.cpp/.h   the vehicle physics demo (beam_sim)
ThrowSim.cpp                 the throw physics engine + dataset generator
ThrowGame.cpp                the game, real physics vs AI prediction live
Terrain.h / ThrowRanges.h    shared C++ headers (terrain shape, input ranges) so
                             the game, the simulator, and dataset gen can't drift apart
Shaders/                     lighting + post-process shaders for throw_game
main.py                      trains the model
features.py                  engineered input features, shared across every script
predict.py                   cold-start CLI single-prediction
predict_server.py            persistent prediction server, what throw_game actually calls
compare_predictions.py       checks the model against real physics + uncertainty calibration
kfold_eval.py                5-fold cross validation
load_dataset.py              sanity-checks the CSV loads into tensors correctly
Throwsxx.py                  quick random-throw inference demo
models/                      model class definition and saved weights
IMPROVEMENTS.md              full history of what was tried, what worked, what didn't
```

## Notes to self / possible next steps

- claw back the lost `0.2968`/`1.871` result - now that the dataset-size confound (700k vs the 1M it was originally trained on) is fixed, try more training time and/or a longer patience before concluding the gap is unreachable
- retry the `physics_baseline_x` feature (`features.py`, currently unused) in isolation - the run that tested it also had an unrelated patience regression, so it was never cleanly evaluated
- `main.py` now checkpoints on every validation improvement, not just at the end - keep that; it's the only thing that saved this project from a second lost-checkpoint incident
- multiple object types with different physics feels
- try scaling the dataset up further and see where the returns actually flatten out
- the model got wide and deep (9 layers, ~7.76M params) mostly through trial and error - a real hyperparameter sweep could probably find something smaller that does as well

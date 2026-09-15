# BeamSim3D

A C++ physics engine that generates synthetic projectile-throw data, a PyTorch neural network trained to predict where things land, and a 3D game that puts the trained model up against the real physics live, with scoring and a player-vs-AI mode.

## What's in here

Three parts, in order of how they build on each other.

**The C++ physics engine** (`ThrowSim.cpp`) simulates an object thrown with a given velocity and mass, integrated frame by frame with gravity, drag, and a bounce off rolling-hill terrain. It has a batch mode that fires hundreds of thousands of randomized throws and logs `vx0, vy0, mass -> final_x, final_y, maxHeight`, and a single-throw mode for asking the real physics for ground truth on one specific input.

**The ML pipeline** (`main.py` and friends) trains a neural network on that dataset to predict landing position and max height from the three launch parameters, then checks the result against the real engine rather than trusting the training loss number.

**The game** (`ThrowGame.cpp`) renders both the real throw and the AI's predicted landing live in a 3D window (raylib, custom lighting shaders, particle effects), lets you play against the AI (guess the landing spot, closest wins), and tracks target-zone scoring and stats across sessions.

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
- `throw_sim` - headless data generator, also has a single-throw ground-truth mode
- `throw_game` - the actual game

Run `./throw_sim` first to generate a dataset, then `./throw_game` to play (needs a trained model - see below - or it'll just show the real physics with the AI prediction marked as unavailable).

`throw_sim` usage:
```
./throw_sim                          # batch mode, writes throw_results.csv
./throw_sim <vx> <vy> <mass>          # single throw, prints final_x,final_y,maxHeight
./throw_sim --trajectory <vx> <vy> <mass>   # prints every simulation step, used to animate the throw in-game
```

One thing that bit me more than once: editing `throwCount` (or anything else) in `ThrowSim.cpp` does nothing on its own. You have to rebuild (`make throw_sim`) AND actually rerun it (`./throw_sim`) or `throw_results.csv` stays exactly as it was. Lost real time chasing "why isn't more data helping" before realizing I was retraining on a stale file more than once.

## Playing the game

```
cd build
./throw_game
```

Controls: arrow keys adjust throw velocity, `[`/`]` adjust mass, `A`/`D` move your landing guess (blue disc), `SPACE` throws, `R` replays the last throw. A magenta flag marks a random target zone each throw - land close to it for points. Your guess vs. the AI's prediction, whoever's closer to the real landing spot wins that round. Stats persist across sessions in `game_stats.txt` at the repo root.

Needs your Python venv active in the same terminal (see below) for the AI prediction to work - without it, the game still runs and shows real physics, just with the AI side marked unavailable instead of crashing.

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

Loads `build/throw_results.csv`, splits it 70/15/15 into train/val/test, normalizes inputs (train-set stats only, no leakage), trains with mini-batches, a learning-rate scheduler, and early stopping. Val loss drives early stopping; test loss is reported separately at the end since it's never touched during training - an honest number instead of one cherry-picked by early stopping. Saves weights + normalization stats to `models/ImprovedNeuralNetwork.pth`. Uses Apple Silicon's MPS backend automatically if available, falls back to CPU otherwise.

Check the model against real physics (not just the training loss number):

```
python3 compare_predictions.py
```

Runs 30 fixed unseen throws through both `throw_sim` and the trained model, prints them side by side, reports mean absolute error per output.

More solid accuracy number than one train/val split:

```
python3 kfold_eval.py
```

5-fold cross validation, mirrors `main.py`'s training setup so the numbers are comparable. Takes a while - it's training 5 separate models.

Quick single prediction from the command line:

```
python3 predict.py <vx> <vy> <mass>
```

Prints `final_x,final_y,maxHeight`. This is what `throw_game` calls under the hood.

## How the physics data is generated

Each throw starts at a fixed height with a random horizontal velocity, vertical velocity, and mass, integrated with gravity and linear drag. It lands on rolling-hill terrain (two sine waves added together) instead of flat ground, bounces with some energy loss, and settles once it's basically stopped moving. `final_x`, `final_y` (resting height, which varies with the terrain), and `maxHeight` (peak height reached mid-flight) all get logged as the targets the network has to predict.

## The model

Feedforward network (`models/ImprovedNeuralNetwork.py`), currently 8 hidden layers. Takes 4 inputs (the 3 raw launch parameters plus one physics-informed engineered feature - the asymptotic drag-decay range, computed in `features.py` and shared by every script that touches the model so it can't drift out of sync). Outputs 3 values: `final_x, final_y, maxHeight`.

Current best result, trained on 500,000 throws:

- Val loss 0.4262, test loss 0.4321 (held-out set never touched during training - close agreement to val means this generalizes, not a lucky split)
- Mean absolute error against real physics on 30 unseen throws: x = 0.969, y = 0.168, max_height = 0.640

`IMPROVEMENTS.md` has the full history of what actually moved these numbers, including the things that didn't work and got reverted.

## Repo layout

```
main.cpp / Physics.cpp/.h   the vehicle physics demo (beam_sim)
ThrowSim.cpp                 the throw physics engine + dataset generator
ThrowGame.cpp                the game, real physics vs AI prediction live
Terrain.h / ThrowRanges.h    shared C++ headers (terrain shape, input ranges) so
                             the game, the simulator, and dataset gen can't drift apart
Shaders/                     lighting + post-process shaders for throw_game
main.py                      trains the model
features.py                  the engineered input feature, shared across every script
predict.py                   CLI single-prediction, what throw_game calls
compare_predictions.py       checks the model against real physics directly
kfold_eval.py                5-fold cross validation
load_dataset.py              sanity-checks the CSV loads into tensors correctly
Throwsxx.py                  quick random-throw inference demo
models/                      model class definition and saved weights
IMPROVEMENTS.md              full history of what was tried, what worked, what didn't
```

## Notes to self / possible next steps

- wind and variable object drag as new physics/model inputs - would need dataset regen + retrain, bigger change than anything above
- multiple object types with different physics feels
- try scaling the dataset up further and see where the returns actually flatten out
- the model got wide and deep (up to 8 layers, ~3.7M params) mostly through trial and error - a real hyperparameter sweep could probably find something smaller that does as well

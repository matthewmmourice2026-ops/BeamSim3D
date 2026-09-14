# BeamSim3D

A soft body vehicle physics simulator written in C++ with raylib, plus a small pipeline that generates synthetic throw data and trains a neural network to predict where an object will land.

## What's in here

There are two parts to this project.

**The C++ engine** simulates a vehicle made of nodes and beams (basically a mass-spring system) and renders it in 3D with raylib. It also has a headless mode that fires 10000 randomized throws (random velocity and mass) onto rolling hill terrain and records where each one comes to rest.

**The Python side** takes that dataset, trains a small feedforward neural network on it, and checks whether the network actually learned real physics or just memorized numbers.

## Building the C++ part

You need CMake and raylib installed. On macOS with Homebrew:

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

This builds two executables:

- `beam_sim`, the 3D window that renders the vehicle
- `throw_sim`, a headless tool that runs 10000 simulated throws and writes the results to `throw_results.csv`

Run `./throw_sim` first if you want fresh data, then `./beam_sim` to see the vehicle in the window.

`throw_sim` also has a single-throw mode for checking one specific input against the real physics: `./throw_sim <vx> <vy> <mass>` prints `final_x,final_y` and exits instead of generating a full dataset. `compare_predictions.py` uses this to get ground truth without re-implementing the physics in Python.

One thing that bit me: if you change something in `ThrowSim.cpp` (like the number of throws or the terrain shape), editing the source does nothing on its own. You have to rebuild (`make throw_sim`) AND actually run it again (`./throw_sim`) or `throw_results.csv` stays exactly as it was. I lost a while chasing "why isn't more data helping" before realizing I'd changed the throw count but never regenerated the file, so I was retraining on the same old dataset the whole time.

## The machine learning part

This part needs pandas and PyTorch. I'd recommend using a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
pip install pandas torch
```

Then run the training script:

```
python3 main.py
```

This loads `build/throw_results.csv`, splits it 80/20 into train and validation sets, normalizes the inputs, and trains the network with early stopping so it doesn't just overfit the training data. It saves the trained weights (plus the normalization stats it used) to `models/ImprovedNeuralNetwork.pth`.

To try the model on a new throw you didn't record, run:

```
python3 Throwsxx.py
```

It picks a random velocity and mass and prints what the model predicts.

To actually check the model against the real physics (not just eyeball a number), run:

```
python3 compare_predictions.py
```

This runs 30 fixed unseen throws through both the real C++ engine (via `throw_sim`'s single-throw mode) and the trained model, prints them side by side, and reports mean absolute error. This is the real way to check accuracy, not just trusting the training loss number.

If you want a more solid accuracy number than one train/val split, there's also:

```
python3 kfold_eval.py
```

Does 5-fold cross validation and reports mean and standard deviation of validation loss across folds. Takes a couple minutes to run since it's training 5 separate models.

## How the physics data is generated

Each throw starts at a fixed height and gets a random horizontal velocity, vertical velocity, and mass. It's integrated frame by frame with gravity and a simple drag force, and lands on rolling hill terrain (a couple of sine waves added together) instead of flat ground, so the resting height actually depends on where it lands. It bounces with some energy loss and settles once it's basically stopped moving. The final x and y position is what gets logged as the target the network has to predict.

## The model

It's a plain feedforward network, four linear layers with batch norm and dropout in between. Nothing fancy, this was more about building a full pipeline (simulate data, train on it, verify it against ground truth) than chasing state of the art architecture.

Current numbers, trained on 10000 throws:

- Validation loss: 0.0114
- 5-fold cross validation: mean 0.0105, std dev 0.0013 across folds (tight, so this isn't a lucky split)
- Mean absolute error against the real C++ physics on 30 unseen throws: x = 0.104, y = 0.038

Those errors are small relative to the ranges involved (x lands anywhere from about -80 to 80, y from about -2 to 2), so the network is actually picking up the real relationship, not just memorizing noise.

## Repo layout

```
main.cpp                raylib window, renders the vehicle
Physics.cpp / .h         soft body physics (nodes, beams, wheels, engine)
ThrowSim.cpp             headless simulator that generates throw_results.csv,
                         also has a single-throw CLI mode for ground truth checks
main.py                  trains the neural network, saves weights + normalization stats
load_dataset.py          quick script to sanity check the CSV loads into tensors correctly
Throwsxx.py              runs inference on a random new throw
compare_predictions.py   checks the model against the real physics engine directly
kfold_eval.py            5-fold cross validation for a more solid accuracy number
models/                  model class definition and saved weights
```

## Notes to self / possible next steps

- try scaling the dataset up further and see where the returns actually flatten out
- add a proper terrain-following slide instead of the simplified vertical-only bounce reflection
- the model architecture growth (16 -> 48 -> ... -> 4 layers) was mostly trial and error, could probably get similar accuracy with something smaller if I did a real hyperparameter sweep

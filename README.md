# BeamSim3D

A soft body vehicle physics simulator written in C++ with raylib, plus a small pipeline that generates synthetic throw data and trains a neural network to predict where an object will land.

## What's in here

There are two parts to this project.

**The C++ engine** simulates a vehicle made of nodes and beams (basically a mass-spring system) and renders it in 3D with raylib. It also has a headless mode that fires 1000 randomized throws (random velocity and mass) and records where each one comes to rest.

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
- `throw_sim`, a headless tool that runs 1000 simulated throws and writes the results to `throw_results.csv`

Run `./throw_sim` first if you want fresh data, then `./beam_sim` to see the vehicle in the window.

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

Edit the `new_throw` list in that file to whatever velocity and mass you want to test.

## How the physics data is generated

Each throw starts at a fixed height and gets a random horizontal velocity, vertical velocity, and mass. It's integrated frame by frame with gravity and a simple drag force, bounces off the ground with some energy loss, and settles once it's basically stopped moving. The final x and y position is what gets logged as the target the network has to predict.

## The model

It's a plain feedforward network, three linear layers with batch norm and dropout in between. Nothing fancy, this was more about building a full pipeline (simulate data, train on it, verify it against ground truth) than chasing state of the art architecture.

Last time I checked, validation loss was around 0.05 and a spot check against the actual C++ physics for an unseen throw was off by about 1 unit on the x coordinate. Not perfect but close enough to show the network is picking up the real relationship, not just noise.

## Repo layout

```
main.cpp              raylib window, renders the vehicle
Physics.cpp / .h       soft body physics (nodes, beams, wheels, engine)
ThrowSim.cpp           headless simulator that generates throw_results.csv
main.py                trains the neural network
load_dataset.py        quick script to sanity check the CSV loads into tensors correctly
Throwsxx.py            runs inference on a new, unseen throw
models/                model class definition and saved weights
```

## Notes to self / possible next steps

- try adding terrain height variation so the network has to learn something less trivial than "land on flat ground"
- k-fold cross validation instead of one train/val split
- clean up the leftover files from earlier experiments still sitting in the repo root

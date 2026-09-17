# race_game

A C++ racing mode where a population of AI cars learns to drive purely through a **genetic algorithm** — no gradients, no PyTorch, no subprocess. Simulate a generation, score by track progress, keep the best brain, mutate the rest, repeat. Companion to `throw_game` (see the root `README.md`/`IMPROVEMENTS.md`), but a completely different AI approach: neuroevolution instead of supervised regression.

## What's in here

```
Track.h        closed-loop spline track (build, nearest-point/raycast queries, ribbon mesh)
CarPhysics.h   simple kinematic car model (no forces/mass/slip)
CarBrain.h     the "brain" - a tiny MLP + mutate/crossover/save/load
RaceGame.cpp   the executable: genetic algorithm, obstacles, upgrades, rendering, HUD
```

`Track.h` builds a closed-loop Catmull-Rom spline through ~12 hand-placed control points, resampled once at startup into a fixed arc-length table (`ds=1.0`, `halfWidth=6.0`). Everything else — the car's progress-along-track fitness metric, its 7 sensor raycasts, off-track detection — reads from that table via windowed searches instead of scanning the whole track every frame. It's a **single closed loop**: no branching/forking paths, by design (see "Not done" below).

`CarPhysics.h` is deliberately not a rigid-body model: direct integration of speed and heading, no drift/slip angle. Corners a little robotically compared to a real car, which is fine for a genetic-algorithm testbed you want to debug by inspection. `updateCar()` takes an `enginePerf`/`turnPerf` multiplier pair so engine/tire upgrades and collision damage scale speed/accel and turn rate independently — an engine upgrade doesn't silently also improve handling.

`CarBrain.h`'s `Brain` is a flat, padding-free POD struct (`static_assert`-guarded) so mutation/crossover/file save-load are all trivial `float[]` operations, no per-layer serialization:

- **Inputs (11):** 7 sensor raycasts at `{-60,-40,-20,0,20,40,60}` degrees relative to heading (each the *minimum* of distance-to-track-edge and distance-to-nearest-obstacle, so one ray sees both without a separate obstacle-sensing architecture), normalized speed, the car's own previous steering/throttle output fed back in (short-term memory — it can react to what it was just doing, not just the current frame in isolation), and the distance to the nearest *moving* traffic obstacle specifically (so traffic, which can be timed/waited out, isn't confused with a wall, which can't).
- **Hidden layer:** 16 units, `tanh` activations.
- **Outputs (6):** steering, throttle, then 4 purchase-preference scores (engine/tires/armor/"keep setup" pack) — the same evolved network decides both how to drive and what to spend its earnings on. Highest-scoring *affordable* option wins each frame it can afford one.

## Building and running

Same CMake project as `throw_game` — nothing extra to install.

```
cd build
cmake ..
make race_game
./race_game
```

Auto-loads `../race_brain.txt` if present and resumes from that generation; otherwise starts a fresh population of random brains at generation 0.

## Controls

```
ESC                   pause (does not quit - window close button does)
TAB                   settings panel (mouse sensitivity, invert Y, generation end mode/limit)
G                     garage panel (NFS-style: each car's coins, tier costs, last AI purchase)
1-6                   pin camera to a specific car
0                     auto-follow the current fitness leader
right-drag / scroll   orbit / zoom camera
M                     (in settings) toggle generation-end mode: time limit vs. lap count
[ / ]                 (in settings) adjust the time limit or lap target
F11                   fullscreen
```

## How a generation works

6 cars (`POP_SIZE`) start on a grid each generation. Every frame, each alive car: casts its 7 sensors, runs its brain forward, drives, and its arc-length progress along the track (unwrapped across the lap seam) accumulates into `fitness`. A generation ends when every car has died, the configured time/lap limit is hit, or a 120s safety cap (`SAFETY_MAX_TIME`) trips regardless of settings.

At generation end (`selectAndMutate`): sort by fitness, slot 0 keeps the best brain unchanged (elitism), slot 1 is a mutated crossover of the top two, slots 2-5 are mutated clones of the best (`MUTATION_RATE=0.15`, `MUTATION_STRENGTH=0.3` — per-weight probability and gaussian-noise scale). The winning brain and generation number are saved to `race_brain.txt` every generation, so killing and relaunching `./race_game` resumes instead of restarting cold.

## Health, damage, and collisions

Cars don't die on first contact. Hitting a wall or obstacle bounces the car (reflects velocity off the surface normal, damps to `WALL_RESTITUTION=0.35`, pushes back in from the boundary by a margin — clamping exactly onto the boundary was measured to re-trigger a bounce almost every frame) and drains `HEALTH_DAMAGE_PER_HIT=20` from a `HEALTH_MAX=100` pool — 5 solid hits before a car actually stops updating for the rest of the generation. Current health also scales `enginePerf`/`turnPerf` down (`0.5 + 0.5*health/healthMax`), so a damaged car drives worse well before it dies, not right up until it does.

Every wall hit also costs `WALL_HIT_PENALTY=15` fitness — large enough relative to typical per-lap progress that bouncing off walls for cheap distance is a losing strategy, not a shortcut (this was tuned up from an earlier value of `1.0`, which cars were exploiting). The wall itself is a visible red/white striped barrier mesh rendered along the track edges, not just an invisible collision boundary, with a legend line in the HUD spelling out the exact penalty.

## Obstacles and traffic

4 static obstacles and 2 slow lateral-patrolling "traffic" obstacles sit on the track, both detected through the same 7 sensor rays used for track edges (`raycastObstacleDistance`, taken as the `min` alongside edge distance — no separate brain-architecture change needed for cars to "see" them). Collision uses the same bounce/health-damage consequence as a wall hit.

## Upgrades and the economy

Cars earn coins from laps (`COINS_PER_LAP=50`) and distance traveled (`COINS_PER_DISTANCE_TICK=5` per `50` units). 3 tiers each of engine (`+15%` speed/accel/tier, stacking), tires (`+15%` turn rate/tier), and armor (`+25` max health/tier) upgrades, priced 60-280 coins. The evolved brain itself picks which category to buy — highest-scoring affordable option among its own 4 purchase outputs, same fitness pressure driving both "drive well" and "spend well."

A brain can also spend 250 coins on a one-shot **Keep Setup Pack**, which carries that car *slot's* tier levels into the next generation instead of resetting to stock. Important caveat, documented in-code and in the Garage panel: persistence is by slot, not by brain identity — the brain occupying that slot next generation is a mutated descendant via `selectAndMutate`, not literally the same brain that earned the parts.

## Graphics

Same custom Blinn-Phong lighting shader as `throw_game` (`Shaders/lighting.vs`/`.fs`, one warm directional light + ambient + distance fog), now with texture sampling added (no-op for any model without an assigned texture, so `throw_game` is unaffected). Procedurally generated grass and road textures (`GenImagePerlinNoise`, tinted/contrasted, no external asset files) on a new ground-plane mesh and the track ribbon respectively — texcoords tile every few units instead of stretching one texture over the whole track length. Internal render resolution is 1280x800 with HUD font sizes/panel layout sized to match.

## Persistence and logging

```
race_brain.txt              latest best brain + generation number (repo root, survives build/ wipes)
race_settings.txt           camera/settings persistence
race_logs/summary.csv       one row per generation: every car's fitness/distance/laps/wallHits/health/died
race_logs/gen_NNNNNN_actions.csv   every car, every frame of that generation: steering/throttle/speed/health/coins/upgrade
race_logs/gen_NNNNNN_brain.txt     that generation's winning brain weights (distinct from race_brain.txt's single latest-best)
```

## Not done / deferred

- **Branching track paths.** `Track.h`'s nearest-point search, raycasting, and lap-counting all assume a single 1D closed loop. Supporting a fork/rejoin or dead-end branch (so the AI can be observed choosing between routes) needs a graph-based redesign, not a quick addition — flagged, not started.
- Cars are simple multi-box models (body/cabin/4 wheels via `DrawCube`, rotated by heading), not an imported 3D asset.

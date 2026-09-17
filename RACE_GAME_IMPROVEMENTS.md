# race_game improvements log

Chronological history of what was built, changed, and reversed in `race_game`, in the same spirit as the root `IMPROVEMENTS.md` (which covers the ML/`throw_game` side) — what was tried, what worked, what didn't, with real numbers where there are real numbers to give. See `RACE_GAME_README.md` for the current architecture reference; this file is the "how it got there."

## 1. Design decision: genetic algorithm over PPO

The brief was 6 AI cars that "self-improve from being last to get number 1." Full policy-gradient RL (PPO) was the "textbook" choice and was explicitly considered and rejected in favor of neuroevolution: PPO is heavier and far easier to get subtly wrong (advantage estimation, clipping, value-function coupling), and a genetic algorithm maps directly onto the stated goal — simulate a generation, score by track progress, keep the best, mutate into the next generation — with no gradients to debug. Mutation is `memcpy` + gaussian noise on a flat weight array; a 6-car MLP forward pass is a few hundred multiply-adds per frame, trivially fast in-process. This also meant zero Python/IPC: unlike `throw_game`'s `predict_server.py` bridge (background thread, `fork`/`exec`, blocking round-trips), the whole feature is pure C++.

## 2. Initial build

`Track.h`: closed-loop Catmull-Rom spline, ~12 hand-placed control points, resampled to a fixed arc-length table. Built in a windowed nearest-point/raycast search (not a full scan per frame) and a build-time sanity check that warns if a corner's control points pinch the track width below `2*halfWidth*0.9` — catches a bad control-point layout immediately instead of shipping a silently-broken collision surface. `dt` is clamped to `min(GetFrameTime(), 1/30)` before driving `updateCar()`, so a frame-time spike (e.g. right after unpausing) can't move a car far enough in one step to escape the search window.

`CarBrain.h`: `IN=8` (7 sensor rays + normalized speed), `HID=10`, `OUT=2` (steering, throttle) at this point. `CarPhysics.h`: kinematic, no forces/mass/slip — cars corner instantly with no drift, a deliberate simplification for a testbed meant to be debuggable by inspection.

6-car population, elitism (best brain carries over unchanged) + mutated crossover of the top two + mutated clones of the best for the rest, `race_brain.txt` persistence (same plain-text convention as `game_stats.txt`) so killing and relaunching resumes instead of restarting cold. Camera, pause, and HUD panel/text-outline helpers reimplemented locally in `RaceGame.cpp` following `ThrowGame.cpp`'s patterns (not shared code, to avoid touching the working throw game).

## 3. Reversed: "never dies" → health-based death, damage, and an upgrade economy

The first build had cars bounce off walls forever, never actually dying — cheap to debug, but not what was asked for. User feedback reversed this explicitly: cars should die when a health bar runs out (not instantly on contact), damage should degrade driving performance as health drops, and cars should be customizable — laps/distance unlock upgrade currency, and the *AI itself* (not a hardcoded rule) decides which part to buy.

This is the origin of the current health system (`HEALTH_MAX=100`, `HEALTH_DAMAGE_PER_HIT=20` — 5 hits to die), the `enginePerf`/`turnPerf` damage-scaling in `CarPhysics.h::updateCar()`, and the whole upgrade economy: coins from laps/distance, 3 tiers each of engine/tire/armor parts, and `Brain::OUT` growing `2→5` to add 3 purchase-preference outputs so the evolved network — not a separate policy — makes the spending decision, under the same fitness pressure as driving skill.

## 4. Rebound physics, configurable session length, full logging

Alongside the health rework: wall contact became a real bounce (reflect velocity off the boundary normal, damp by `WALL_RESTITUTION=0.35`, push back in by a margin) instead of a snap-to-boundary — clamping exactly onto the boundary was measured to bounce a car nearly every single frame (thousands of hits/generation) because a heavily-damped near-zero-speed bounce sitting right at the edge gets pushed straight back out by the very next frame's steering output. The margin plus a minimum post-bounce speed (`MIN_BOUNCE_SPEED`) fixed it. User-configurable generation-end mode was added (time limit or lap count, `TAB` settings panel, `M`/`[`/`]`), plus the full `race_logs/` directory (per-generation `summary.csv`, per-frame `actions.csv`, per-generation brain snapshot) so a long run leaves a complete, inspectable trail instead of only the single latest-best `race_brain.txt`.

## 5. Per-action logging, graphics overhaul, Garage menu

Extended `race_logs/gen_NNNNNN_actions.csv` to log every car's action every frame (steering, throttle, speed, health, coins, upgrade purchased). Graphics brought in line with `throw_game`: the same custom Blinn-Phong lighting shader (`Shaders/lighting.vs`/`.fs`), collision spark particles, fixed-resolution render-texture + letterboxed compositing. Added the Garage panel (`G` key) — an NFS-style shop screen showing each car's coins, tier/next-cost per category, and its last AI-made purchase, so "what is the AI actually buying" is visible instead of implicit in a CSV.

## 6. Reward-shaping fix: cars were bouncing for cheap progress, not learning to avoid walls

User feedback: cars needed to "learn the directions, not just crash into the walls and hope for the best." Diagnosis: `WALL_HIT_PENALTY` was `1.0`, negligible next to the hundreds of fitness points available from distance progress — a car could bounce off a wall repeatedly and still come out ahead as long as it kept moving forward on net. Raised to `15.0` (documented as "roughly half a lap's worth of progress per hit"). Verified with a direct math check against a real `race_logs/summary.csv` row before deploying further changes: `39 distance - 5 hits*15 = -36 fitness` — wall-hopping now reliably produces a negative score instead of a positive one.

## 7. Obstacles and traffic

Added `struct Obstacle` (static pillars + slow lateral-patrolling "traffic," `TRAFFIC_PATROL_SPEED=0.22` patrol-units/sec) and `raycastObstacleDistance` — a closed-form 2D ray-vs-circle check, combined with the existing edge-distance raycast via `fmin` in the same 7 sensor rays. Deliberately did **not** add dedicated obstacle-sensing inputs to `Brain::IN` at this stage — the existing sensors just got more to look at, no architecture change needed for cars to react to obstacles at all (see section 11 for later, deliberate sensor additions). Collision reuses the same bounce/health-damage consequence as a wall hit via a shared `bounceCarState()` helper (refactored out of what was previously wall-only inline code).

One self-caught bug during this pass: obstacle cylinders were first drawn with their *center* passed as `DrawCylinder`'s position argument (`{obs.pos.x, 1.6f, obs.pos.z}`), which would have floated them in mid-air — `DrawCylinder`'s position is the base, not a center reference. Caught before building, corrected to `{obs.pos.x, Track::trackY, obs.pos.z}`.

## 8. "Keep Setup Pack" persistence

User asked for a way to buy a specific pack so upgrades from the *last* race don't reset. Implemented as a 4th brain-purchasable option (`Brain::OUT` `5→6`), one-shot, `250` coins (`PERSIST_PACK_COST`): if bought, that car *slot's* engine/tire/armor tiers carry into the next generation's starting state instead of resetting to stock. Explicitly documented caveat, in code and in the Garage panel text: persistence is by slot, not by brain identity — the brain occupying that slot next generation is a mutated descendant via `selectAndMutate`, not literally the same brain that earned the parts. This is consistent with how every other per-generation reset already works, not a special case.

## 9. Resolution and HUD typography pass

User feedback: the HUD "seems like 2001," asked to either modernize it or at least sharpen the existing retro look. Internal render resolution bumped `1000x600 → 1280x800`; every HUD font size and panel width/position was hardcoded and did not auto-scale, so all of it was rescaled by hand (roughly: 13px→16px, 14px→17px, 15px→18px, 17px→20px, 20px→24px, 44px→56px; panel widths like the leaderboard `boardW` 270→350, settings `sw` 292→360, garage `gw` 760→980 with its 6 column offsets rescaled to match). Decided against a custom font asset (none exists in the repo, no external asset fetching) — kept raylib's default bitmap font at larger point sizes, per the user's explicit fallback ("stick with this retro game typography but improve the... resolutions").

## 10. Procedural textures and a visible ground plane

Added grass and road textures via `GenImagePerlinNoise` (tinted/contrasted, no external asset files, same no-fetching convention as `main.cpp`'s beam_sim fallback) and a new flat ground-plane mesh (`buildGroundPlane`) sized to the track's bounding box plus a 60-unit margin, since no ground geometry existed before (the track mesh itself is only the road ribbon).

This required a real change to the *shared* lighting shader: `Shaders/lighting.fs` had never sampled `texture0` at all — it only multiplied vertex color by `colDiffuse`. Added `vec3 texel = texture(texture0, fragTexCoord).rgb;` into the albedo term. Because `throw_game`'s terrain/ball never assign a diffuse texture, their material's texture stays raylib's default 1x1 white pixel, making this sampling a no-op for them — verified by rebuilding `throw_game` after the change and confirming a clean build. `Track.h`'s texcoords also changed from `u = i/count` (one texture tile stretched across the entire track loop, which would have rendered as a single blurred smear) to `u = (i*ds)/4.0` so the road texture repeats every ~4 units and actually reads as asphalt grain.

## 11. Visible track-edge walls

User feedback: wanted a physical, visible wall along the track so the existing wall-hit health/fitness penalty (sections 3, 6) is obvious, not implicit. Added `buildWallModel()` — a red/white striped vertical ribbon mesh along `Track::leftEdge`/`rightEdge` (`WALL_HEIGHT=1.4`, restriped every `WALL_STRIPE_SAMPLES=6` samples), drawn with backface culling disabled for the draw call so it reads correctly from both the track side and the outside regardless of triangle winding, rather than trying to hand-tune winding order per edge. Purely visual — collision still happens against the same `leftEdge`/`rightEdge` arrays via the existing raycasts, this mesh doesn't touch physics. Added a red legend line to the top HUD panel spelling out the exact penalty (`Hit red/white wall: -N fitness, -N HP`, pulling the live constants rather than hardcoding a string that could drift out of sync).

## 12. Brain "reasoning" upgrade: memory and dedicated traffic sensing

User asked to improve the AI's ability to "understand and reason," not just react. Two additions to `Brain::IN` (`8→11`): the car's own previous steering/throttle output fed back in as inputs (short-term memory — an Elman-style feedback loop, not a full RNN, but enough for the network to condition on what it was just doing instead of treating every frame as independent), and a dedicated nearest-moving-traffic-distance input (a direct distance query over `Obstacle::moving` obstacles, not ray-based) so traffic — which can be timed or waited out — isn't lumped in with a wall, which can't. `Brain::HID` widened `10→16` for the extra inputs' worth of representational capacity.

This is an architecture change: `BRAIN_WEIGHT_COUNT` changes, and `loadBrainFile`'s `weightCount` guard correctly rejects the old `race_brain.txt` on the first run after this change, starting fresh from generation 0 rather than silently misreading incompatible weights — the same format-version-guard behavior documented in the original implementation plan, exercised for real for the first time here (the earlier `Brain::OUT` 2→5→6 changes hit the same guard).

## Not done / deferred

- **Branching track paths** ("add some other streets to see if the AI will take them or not"). `Track.h`'s nearest-point search, raycasting, and lap-counting-via-arc-length-wrap all assume a single 1D closed loop; a fork/rejoin or dead-end branch needs a graph-based redesign of `Track.h`, not a quick addition. Flagged as a larger architecture change and explicitly not started — worth scoping as its own piece of work rather than bolting on hastily.

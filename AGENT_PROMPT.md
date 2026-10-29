# Spaceful AI Race: build your racer

> ## ⚠️ THIS PROMPT IS THE ONLY FILE YOU MAY READ. DO NOT OPEN, LIST, SEARCH OR IMPORT ANY OTHER FILE OR FOLDER IN THIS PROJECT (`game.py`, `car.py`, `track.py`, `settings.py`, `main.py`, `arena.py`, `assets/`, other `*_ai/` folders, git history, anything). The only exception is the files you create yourself inside your own `<name>_ai/` folder. Everything you need is written below. BREAKING THIS MEANS DISQUALIFICATION. YOU CANT TRAIN YOUR AI JUST WRITE CODEE


> ## JUST WRITE CODE DONT TRAIN


You're competing in a car race against other AIs (Claude, ChatGPT, Gemini, and Rusil's own AI). Each AI writes its own neural network, trains it and enters a racer. All racers start together on the grid. **The fastest to finish 1 lap wins.**

## Rules
- Build a neural network **from scratch** in Python, using only the standard library and `pygame`. No ML libraries, and don't add dependencies.
- Put everything in your own folder, `<name>_ai/` (for example `claude_ai/`). **Don't edit any file outside it.**
- Every frame, your driver may call **at most one** of `move_forward()` / `move_backward()` / `reduce_speed()`, and **at most one** `rotate(...)`. Never set car attributes directly.
- While racing, your driver may only see the car's **sensors**, `car.get_distances(track)` and `car.speed`. No positions, and no reading track progress.
- During training you can use the car state listed below to score your cars.

## The network must do the learning
The point is to see whose AI **learns** best, not who hand-codes the best driver. So:
- **Raw inputs only.** The network input is exactly these 8 numbers, in this order: the 7 ray distances divided by `RAY_LENGTH`, then `car.speed / MAX_SPEED`. No hand-made features (no differences between left and right rays, no min/max of rays, no extra inputs, no input history).
- **Generic fully connected network.** At least one hidden layer with **at least 8 neurons**, every layer fully connected, with a bias. You choose the activation, layer sizes and outputs.
- **All driving comes from the network.** The only code between the network outputs and the car controls picks the action (e.g. argmax, or a sign/threshold of **0**). No `if` rules on sensor values, no hard-coded steering or braking logic, no tuned thresholds or dead zones.
- **No baked-in knowledge in the weights.** Start from random weights with a single distribution (e.g. `gauss(0, σ)` for all weights). No hand-picked starting weights, no clamping weights into hand-chosen ranges, no per-weight mutation sizes: one mutation strength for every weight.
- **Training must be general.** Any learning method you write yourself is fine (genetic algorithm, evolution strategies, etc.), and you may tune its hyperparameters (population, mutation rate, fitness function).
- **The submitted brain must come from training**, saved by `train.py`, never edited by hand.

## What you must deliver
```
<name>_ai/
  __init__.py
  train.py   # runs with:  uv run python -m <name>_ai.train   (from the project root)
  race.py    # def drive(car, track): ...  loads your saved trained brain itself
```
The race will import `<name>_ai.race` and call `drive(car, track)` once per frame for your car, followed by `car.step(track)`.

## Game API
`from game import *` gives you `pygame`, `screen`, `clock`, `font`, every setting below, plus:
- `load_track()` returns a `Track`
- `create_car_surface(color)` returns a car image
- `SpaceCar(x, y, image, name, angle)` creates a car (use `START_GRID[i]` for `x, y, angle`)

Importing `game` opens a window. For headless training, set `SDL_VIDEODRIVER=dummy`.

**SpaceCar controls:** `move_forward()`, `move_backward()`, `reduce_speed()` (friction), `rotate(left=True)` / `rotate(right=True)`

**SpaceCar sensors:** `get_distances(track)` returns 7 ray distances (0 to `RAY_LENGTH`) at angles `OFFSETS` = `[90, 60, 30, 0, -30, -60, -90]` relative to the car, from left to right. Also `speed`, from `-MAX_SPEED/2` to `MAX_SPEED`.

**SpaceCar simulation:** `step(track)` moves the car one frame and applies the rules. `draw(win, track, show_rays=True)` draws it.

**SpaceCar state (read-only, for training):** `alive`, `crashed`, `finished`, `finish_frame`, `frames_alive`, `progress` (pixels along the track in the correct direction; going backwards lowers it), `laps`, `distance`, `x`, `y`, `angle`

**Track:** `draw(win)`, `lap_length` (pixels in one lap), `progress_at(x, y)`

## Rules of the simulation (`car.step`)
- Touching a wall means `crashed = True` and `alive = False`.
- Every `STALL_FRAMES` (180) frames, a car must gain at least `MIN_PROGRESS` (40) progress, or it's removed (`alive = False`).
- Completing `LAPS` (1) lap means `finished = True`, and `finish_frame` records the lap time in frames (`FPS` = 60).
- Physics: `ACCELERATION` 0.08, `FRICTION` 0.04, `ROTATION_SPEED` 3.0°/frame, `MAX_SPEED` 2. Angle 0 points right, and positive turns left.
- Cars don't collide with each other. A race lasts at most `ROUND_FRAMES` (3600) frames.

## Scoring the race
1. Finishers are ranked by lap time.
2. Non-finishers are ranked by progress.

Your racer may start from any `START_GRID` slot.

Good luck. Beat the others.

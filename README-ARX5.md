# ARX5 + pi05_base

This document records the ARX5 runtime that was added to Evo-RL for running LeRobot `pi05_base` directly on the robot.

## 1. Environment setup

Use the repository Conda environment first:

```bash
cd /home/user/workspace/whs/Evo-RL
conda activate lerobot
```

If `lerobot` is not available, use the fallback environment noted in `Evo-RL/AGENTS.md`.

Install the ARX5 runtime extras:

```bash
pip install -r requirements-arx5.txt
```

Point LeRobot to the local ARX SDK checkout:

```bash
export ARX_SDK_ROOT=~/workspace/ARX_X5/py/arx_x5_python
```

The ARX5 SDK itself is not installed by `pip`; it is loaded from `ARX_SDK_ROOT`.

## 2. Camera check

List RealSense devices:

```bash
lerobot-arx5-infer --list-cameras
```

The ARX5 runtime expects the same local camera names used in the dexbotic scripts:

- `side` -> LeRobot `base_0_rgb`
- `wrist` -> LeRobot `left_wrist_0_rgb`
- `front` -> LeRobot `right_wrist_0_rgb`

## 3. Optional dry run

Before touching the real arm, run a stub-only smoke test that still loads the policy stack and saves one inferred chunk:

```bash
lerobot-arx5-infer \
  --task "place shoes on rack" \
  --policy-path lerobot/pi05_base \
  --use-stub \
  --safe-mode \
  --record-dir ./tmp/arx5_stub_record \
  --cameras side:254522071216 wrist:150622073629 front:409122272986
```

If you do not want the script to access hardware cameras during the dry run, replace the three RealSense entries with USB camera indices and add `--use-usb-cams`.

## 4. Real robot inference

The command below mirrors the dexbotic deployment style but runs LeRobot `pi05_base` locally:

```bash
cd /home/user/workspace/whs/Evo-RL
conda activate lerobot
export ARX_SDK_ROOT=~/workspace/ARX_X5/py/arx_x5_python

lerobot-arx5-infer \
  --task "place shoes on rack" \
  --policy-path lerobot/pi05_base \
  --cameras side:254522071216 wrist:150622073629 front:409122272986 \
  --flip-cameras wrist front \
  --duration 0.1 \
  --safe-mode
```

Notes:

- `--duration 0.1` matches the 10 Hz chunk execution used in the dexbotic Pi0.5 client.
- `--safe-mode` gates each predicted chunk behind keyboard confirmation and clips each Cartesian translation step to `0.05 m` by default.
- The runtime uses the bundled quantile stats file `src/lerobot/robots/arx5_follower/pi05_arx5_default_stats.json` unless `--stats-path` is provided.

## 5. Keyboard controls

When keyboard control is enabled (the default):

- `Space`: hold current pose immediately
- `H`: go home, then hold
- `B`: enter teach mode / gravity compensation
- `N`: save the current teach pose
- `M`: move back to the saved teach pose
- `R`: resume policy control
- `I`: in safe mode, request the next predicted chunk
- `Q`: quit

The script starts in a stopped state when keyboard control is enabled. Press `R` first, then `I` for each chunk when `--safe-mode` is active.

## 6. Saving each round to disk (`--record-dir`)

With `--record-dir`, the runtime still runs **policy inference** and writes one folder per round, but it **does not send motion commands** to the arms (same idea as `lerobot_arx5_infer`: save then skip execution). Omit `--record-dir` to run the normal loop that **executes** the predicted chunk on hardware.

Example:

```bash
lerobot-arx5-infer \
  --task "Stack all the paper cups on the table together" \
  --policy-path checkpoints/multi_cups_test0/pretrained_model \
  --cameras front:335122271555 wrist:409122272986 \
  --safe-mode \
  --record-dir /home/user/workspace/whs/Evo-RL/inference_records
```

Each chunk is written under `./inference_records/round_XXXX/` with:

- `input/base.png`
- `input/right_wrist.png`
- `input/current_ee_state.json`
- `output/actions.json`

For dual-arm, `--record-dir` alone avoids `send_joint`; you can add `--use-stub` if you also want the CAN stack stubbed during state reads (see §3).

The dual-arm runner `python -m lerobot.scripts.lerobot_arx5_dual_infer` writes the same layout with filenames derived from the checkpoint’s `observation.images.*` keys (e.g. `dual_towel` uses `input/base.png`, `input/left_wrist.png`, `input/right_wrist.png`), plus `input/state.json` and `output/actions.json`. Pass `--cameras` with those slot names, not `base_0_rgb`-style names, unless your policy was trained with those keys.

For `checkpoints/multi_cups_test0`, the ARX5 runtime resolves the local camera names as:

- `front` -> `observation.images.base`
- `wrist` -> `observation.images.right_wrist`
- `observation.images.empty_camera_0` is left empty and padded by the PI0.5 model

## 7. Useful overrides

Use a local checkpoint instead of the Hub:

```bash
lerobot-arx5-infer \
  --task "place shoes on rack" \
  --policy-path /path/to/local/pi05_base
```

Use a custom stats file:

```bash
lerobot-arx5-infer \
  --task "place shoes on rack" \
  --policy-path lerobot/pi05_base \
  --stats-path /path/to/arx5_stats.json
```

Shorten each executed chunk:

```bash
lerobot-arx5-infer \
  --task "place shoes on rack" \
  --policy-path lerobot/pi05_base \
  --execution-horizon 10
```


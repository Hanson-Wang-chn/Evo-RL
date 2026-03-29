#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run LeRobot Pi0.5 inference directly on an ARX5 arm."""

import argparse
import atexit
import json
import logging
import queue
import sys
import termios
import threading
import time
import tty
from contextlib import nullcontext
from enum import Enum, auto
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots.arx5_follower import ARX5_REAL_STATE_KEYS, ARX5Follower, ARX5FollowerConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import auto_select_torch_device
from lerobot.configs.policies import PreTrainedConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
logger = logging.getLogger(__name__)

DEFAULT_POLICY_PATH = "lerobot/pi05_base"
DEFAULT_LOCAL_CAMERA_MAP = {
    "base_0_rgb": "side",
    "left_wrist_0_rgb": "wrist",
    "right_wrist_0_rgb": "front",
}
ARX5_MULTI_CUPS_CAMERA_MAP = {
    "base": "front",
    "right_wrist": "wrist",
}
DEFAULT_STATS_PATH = files("lerobot.robots.arx5_follower").joinpath("pi05_arx5_default_stats.json")
OBS_IMAGE_PREFIX = "observation.images."


class LoopState(Enum):
    STOPPED = auto()
    RUNNING = auto()
    TEACHING = auto()


class KeyboardListener:
    """Non-blocking keyboard input for Linux terminals."""

    def __init__(self):
        self._queue: queue.Queue[str] = queue.Queue()
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        atexit.register(self.restore)
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self) -> None:
        while True:
            char = sys.stdin.read(1)
            if char:
                self._queue.put(char.lower())

    def get_key(self) -> str | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def restore(self) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


def _parse_camera_specs(specs: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid camera spec '{spec}', expected name:value")
        name, value = spec.split(":", 1)
        mapping[name.strip()] = value.strip()
    return mapping


def _get_policy_image_keys(policy_cfg: PreTrainedConfig) -> list[str]:
    return [
        key.removeprefix(OBS_IMAGE_PREFIX)
        for key in policy_cfg.input_features
        if key.startswith(OBS_IMAGE_PREFIX)
    ]


def _get_required_policy_camera_keys(policy_cfg: PreTrainedConfig) -> list[str]:
    return [key for key in _get_policy_image_keys(policy_cfg) if not key.startswith("empty_camera_")]


def _resolve_local_camera_map(policy_cfg: PreTrainedConfig) -> dict[str, str]:
    required_policy_keys = _get_required_policy_camera_keys(policy_cfg)
    required_policy_key_set = set(required_policy_keys)

    if required_policy_key_set.issubset(DEFAULT_LOCAL_CAMERA_MAP):
        return {policy_key: DEFAULT_LOCAL_CAMERA_MAP[policy_key] for policy_key in required_policy_keys}

    if required_policy_key_set.issubset(ARX5_MULTI_CUPS_CAMERA_MAP):
        return {policy_key: ARX5_MULTI_CUPS_CAMERA_MAP[policy_key] for policy_key in required_policy_keys}

    raise ValueError(
        "Unsupported ARX5 camera layout for this checkpoint. "
        f"Policy expects {sorted(required_policy_key_set)} but the runtime only knows how to map "
        f"{sorted(DEFAULT_LOCAL_CAMERA_MAP)} or {sorted(ARX5_MULTI_CUPS_CAMERA_MAP)}."
    )


def _list_realsense_cameras() -> None:
    try:
        cameras = RealSenseCamera.find_cameras()
    except Exception as error:
        print(f"Unable to query RealSense cameras: {error}")
        return
    if not cameras:
        print("No RealSense cameras found.")
        return
    print(f"Found {len(cameras)} RealSense camera(s):")
    for camera in cameras:
        default_stream = camera.get("default_stream_profile", {})
        print(
            f"  {camera['name']} serial={camera['id']} "
            f"default={default_stream.get('width')}x{default_stream.get('height')}@{default_stream.get('fps')}"
        )


def _make_camera_configs(
    *,
    camera_specs: dict[str, str],
    local_camera_map: dict[str, str],
    required_policy_camera_keys: list[str],
    use_usb_cams: bool,
    width: int,
    height: int,
    fps: int,
    flipped_local_cameras: set[str],
) -> dict[str, Any]:
    configs = {}
    local_to_policy = {local_name: policy_name for policy_name, local_name in local_camera_map.items()}
    for local_name, source in camera_specs.items():
        if local_name not in local_to_policy:
            raise ValueError(
                f"Unknown local camera '{local_name}'. Expected one of {sorted(local_to_policy)}."
            )
        policy_name = local_to_policy[local_name]
        if policy_name not in required_policy_camera_keys:
            logger.info(
                "Ignoring local camera '%s' because policy does not consume '%s'.",
                local_name,
                policy_name,
            )
            continue
        rotation = 180 if local_name in flipped_local_cameras else 0
        if use_usb_cams:
            configs[policy_name] = OpenCVCameraConfig(
                index_or_path=int(source),
                width=width,
                height=height,
                fps=fps,
                rotation=rotation,
            )
        else:
            configs[policy_name] = RealSenseCameraConfig(
                serial_number_or_name=source,
                width=width,
                height=height,
                fps=fps,
                rotation=rotation,
            )
    missing = set(required_policy_camera_keys) - set(configs)
    if missing:
        raise ValueError(
            "Missing required cameras for this policy: "
            f"{sorted(local_camera_map[name] for name in missing)}"
        )
    return configs


def _load_stats(stats_path: Path) -> dict[str, dict[str, list[float]]]:
    return json.loads(stats_path.read_text(encoding="utf-8"))


def _load_policy_bundle(
    policy_path: str,
    device_override: str | None,
    stats_path: Path | None,
    policy_cfg: PreTrainedConfig | None = None,
):
    if policy_cfg is None:
        logger.info("Loading PI05 policy config from %s", policy_path)
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    else:
        logger.info("Using preloaded PI05 policy config from %s", policy_path)
    if policy_cfg.type != "pi05":
        raise ValueError(f"Expected a pi05 policy, got '{policy_cfg.type}'.")

    target_device = torch.device(device_override) if device_override else auto_select_torch_device()
    policy_cfg.device = str(target_device)
    policy_cfg.pretrained_path = policy_path

    logger.info("Instantiating policy on device=%s", target_device)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=policy_path, config=policy_cfg)
    policy.to(target_device)
    policy.eval()

    if stats_path is None:
        logger.info("Loading pre/post processors from checkpoint: %s", policy_path)
        preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=policy_path)
    else:
        logger.info("Loading normalization stats from %s", stats_path)
        stats = _load_stats(stats_path)
        preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, dataset_stats=stats)
    logger.info("Policy bundle ready.")
    return policy_cfg, policy, preprocessor, postprocessor, target_device


def _build_dataset_features(robot: ARX5Follower) -> dict[str, dict[str, Any]]:
    return combine_feature_dicts(
        hw_to_dataset_features(robot.observation_features, OBS_STR),
        hw_to_dataset_features(robot.action_features, ACTION),
    )


def _predict_action_chunk(
    *,
    robot_observation: dict[str, Any],
    dataset_features: dict[str, dict[str, Any]],
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    robot_type: str,
    execution_horizon: int,
    use_amp: bool,
) -> list[dict[str, float]]:
    observation_frame = build_dataset_frame(dataset_features, robot_observation, prefix=OBS_STR)
    processed_observation = prepare_observation_for_inference(
        dict(observation_frame),
        device,
        task=task,
        robot_type=robot_type,
    )
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        processed_observation = preprocessor(processed_observation)
        action_chunk = policy.predict_action_chunk(processed_observation)
        action_chunk = postprocessor(action_chunk)

    action_names = dataset_features[ACTION]["names"]
    action_chunk = action_chunk.squeeze(0).to("cpu")
    horizon = min(execution_horizon, action_chunk.shape[0])
    actions = []
    for index in range(horizon):
        row = action_chunk[index]
        actions.append({name: float(row[offset]) for offset, name in enumerate(action_names)})
    return actions


def _clip_safe_actions(
    actions: list[dict[str, float]],
    current_pose: list[float],
    max_joint_step: float,
) -> list[dict[str, float]]:
    previous_joint = np.asarray(current_pose[:6], dtype=np.float64)
    safe_actions = []
    for action in actions:
        safe_action = dict(action)
        target_joint = np.asarray([safe_action[key] for key in ARX5_REAL_STATE_KEYS[:6]], dtype=np.float64)
        delta = target_joint - previous_joint
        clipped_delta = np.clip(delta, -max_joint_step, max_joint_step)
        if not np.allclose(delta, clipped_delta):
            target_joint = previous_joint + clipped_delta
            for axis, key in enumerate(ARX5_REAL_STATE_KEYS[:6]):
                safe_action[key] = float(target_joint[axis])
            logger.warning(
                "SAFE MODE: capped joint step, max per-joint delta=%.4f.",
                max_joint_step,
            )
        previous_joint = target_joint
        safe_actions.append(safe_action)
    return safe_actions


def _save_chunk_io(
    *,
    record_dir: Path,
    round_index: int,
    observation: dict[str, Any],
    current_joint_state: list[float],
    actions: list[dict[str, float]],
    camera_names: list[str],
) -> Path:
    round_dir = record_dir / f"round_{round_index:04d}"
    input_dir = round_dir / "input"
    output_dir = round_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for camera_name in camera_names:
        image = observation[camera_name]
        Image.fromarray(image).save(input_dir / f"{camera_name}.png")

    (input_dir / "current_joint_state.json").write_text(
        json.dumps({"current_joint_state": current_joint_state}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "actions.json").write_text(json.dumps(actions, indent=2) + "\n", encoding="utf-8")
    return round_dir


def _log_keyboard_help(safe_mode: bool) -> None:
    base_message = "Keyboard: [Space] stop | [H] home | [B] teach | [N] record pose | [M] goto pose | [R] resume | [Q] quit"
    if safe_mode:
        base_message += " | [I] next chunk"
    logger.info(base_message)


def _run_keyboard_command(
    *,
    key: str | None,
    robot: ARX5Follower,
    state: LoopState,
    safe_mode: bool,
    request_next_chunk: bool,
) -> tuple[LoopState, bool, bool]:
    running = True
    if key == " ":
        robot.hold_position()
        logger.warning("Emergency stop: holding current pose.")
        return LoopState.STOPPED, False, True
    if key == "q":
        logger.info("Quit requested.")
        return state, request_next_chunk, False
    if state == LoopState.STOPPED:
        if key == "h":
            logger.info("Moving ARX5 to the home pose.")
            robot.hold_position()
            time.sleep(0.1)
            robot.go_home()
            time.sleep(2.0)
            robot.hold_position()
        elif key == "b":
            robot.enter_teach_mode()
            logger.info("Teach mode enabled. Drag the arm, then press [N] to save the pose.")
            return LoopState.TEACHING, request_next_chunk, running
        elif key == "m":
            if robot.has_recorded_pose():
                robot.move_to_recorded()
                logger.info("Moved to the recorded pose.")
            else:
                logger.warning("No recorded pose found. Use [B] then [N] first.")
        elif key == "r":
            robot.hold_position()
            time.sleep(0.1)
            logger.info("Resumed policy control.")
            return LoopState.RUNNING, not safe_mode, running
    elif state == LoopState.TEACHING:
        if key == "n":
            recorded_pose_path = robot.exit_teach_mode_and_record()
            logger.info("Recorded pose saved to %s", recorded_pose_path)
            return LoopState.STOPPED, False, running
    elif state == LoopState.RUNNING and safe_mode and key == "i":
        logger.info("SAFE MODE: next chunk requested.")
        return state, True, running
    return state, request_next_chunk, running


def _execute_chunk(
    *,
    robot: ARX5Follower,
    actions: list[dict[str, float]],
    step_duration_s: float,
    keyboard: KeyboardListener | None,
    safe_mode: bool,
) -> tuple[LoopState, bool, bool]:
    state = LoopState.RUNNING
    request_next_chunk = not safe_mode
    running = True
    for action in actions:
        step_start = time.perf_counter()
        if keyboard is not None:
            key = keyboard.get_key()
            state, request_next_chunk, running = _run_keyboard_command(
                key=key,
                robot=robot,
                state=state,
                safe_mode=safe_mode,
                request_next_chunk=request_next_chunk,
            )
            if state != LoopState.RUNNING or not running:
                break
        robot.send_action(action)
        precise_sleep(max(step_duration_s - (time.perf_counter() - step_start), 0.0))

    return state, request_next_chunk, running


def _log_predicted_actions(actions: list[dict[str, float]]) -> None:
    for action_index, action in enumerate(actions):
        logger.debug("Action[%d]: %s", action_index, action)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LeRobot pi05_base directly on an ARX5 arm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, default=None, help="Natural language task instruction.")
    parser.add_argument("--policy-path", type=str, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--policy-device", type=str, default=None)
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help=(
            "Optional normalization stats path. If omitted, load policy_preprocessor/postprocessor "
            "directly from --policy-path."
        ),
    )
    parser.add_argument("--execution-horizon", type=int, default=None)
    parser.add_argument("--duration", type=float, default=0.1, help="Seconds per action step.")

    parser.add_argument("--can-port", type=str, default="can0")
    parser.add_argument("--arm-type", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--use-stub", action="store_true")

    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["side:254522071216", "wrist:150622073629", "front:409122272986"],
        help="Local camera specs as name:serial_or_index using the dexbotic names side/wrist/front.",
    )
    parser.add_argument("--use-usb-cams", action="store_true")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip-cameras", nargs="*", default=[])
    parser.add_argument("--list-cameras", action="store_true")

    parser.add_argument("--safe-mode", action="store_true", default=False)
    parser.add_argument("--max-joint-step", type=float, default=0.02)
    parser.add_argument("--no-keyboard", action="store_true")
    parser.add_argument("--record-dir", type=Path, default=None)
    parser.add_argument("--robot-id", type=str, default="arx5")
    parser.add_argument("--protect-on-disconnect", action="store_true", default=True)
    parser.add_argument("--no-protect-on-disconnect", dest="protect_on_disconnect", action="store_false")

    args = parser.parse_args()

    if args.list_cameras:
        _list_realsense_cameras()
        return
    if not args.task:
        parser.error("--task is required unless --list-cameras is used.")
    if args.safe_mode and args.no_keyboard:
        parser.error("SAFE MODE requires keyboard control.")
    if args.execution_horizon is not None and args.execution_horizon <= 0:
        parser.error("--execution-horizon must be positive.")

    logger.info("Loading policy config from %s", args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    required_policy_camera_keys = _get_required_policy_camera_keys(policy_cfg)
    local_camera_map = _resolve_local_camera_map(policy_cfg)
    camera_specs = _parse_camera_specs(args.cameras)
    camera_configs = _make_camera_configs(
        camera_specs=camera_specs,
        local_camera_map=local_camera_map,
        required_policy_camera_keys=required_policy_camera_keys,
        use_usb_cams=args.use_usb_cams,
        width=args.cam_width,
        height=args.cam_height,
        fps=args.fps,
        flipped_local_cameras=set(args.flip_cameras),
    )

    robot = ARX5Follower(
        ARX5FollowerConfig(
            id=args.robot_id,
            port=args.can_port,
            arm_type=args.arm_type,
            use_stub=args.use_stub,
            cameras=camera_configs,
            protect_on_disconnect=args.protect_on_disconnect,
        )
    )
    policy_cfg, policy, preprocessor, postprocessor, device = _load_policy_bundle(
        policy_path=args.policy_path,
        device_override=args.policy_device,
        stats_path=args.stats_path,
        policy_cfg=policy_cfg,
    )
    dataset_features = _build_dataset_features(robot)
    camera_names = list(robot.cameras)
    execution_horizon = args.execution_horizon or int(policy_cfg.n_action_steps)
    keyboard = None if args.no_keyboard else KeyboardListener()
    state = LoopState.RUNNING if keyboard is None else LoopState.STOPPED
    request_next_chunk = keyboard is None or not args.safe_mode
    round_index = 0
    running = True

    try:
        logger.info("Connecting ARX5 runtime.")
        logger.info("Camera sources: %s", {name: str(source) for name, source in camera_specs.items()})
        logger.info("Resolved local camera map: %s", local_camera_map)
        logger.info("Stub arm: %s", args.use_stub)
        robot.connect()
        logger.info("ARX5 runtime connected. Keyboard enabled: %s", keyboard is not None)
        if keyboard is not None:
            logger.info("Moving ARX5 to the home pose.")
            robot.go_home()
            time.sleep(2.0)
            robot.hold_position()
            _log_keyboard_help(args.safe_mode)
            if args.safe_mode:
                logger.info("SAFE MODE is armed. Press [R] to resume, then [I] for each chunk.")
        else:
            logger.info("Starting continuous ARX5 inference loop without keyboard control.")

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        while running:
            if keyboard is not None:
                previous_state = state
                key = keyboard.get_key()
                state, request_next_chunk, running = _run_keyboard_command(
                    key=key,
                    robot=robot,
                    state=state,
                    safe_mode=args.safe_mode,
                    request_next_chunk=request_next_chunk,
                )
                if previous_state != LoopState.RUNNING and state == LoopState.RUNNING:
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()
            if not running:
                break
            if state != LoopState.RUNNING:
                time.sleep(0.05)
                continue
            if args.safe_mode and not request_next_chunk:
                time.sleep(0.05)
                continue

            observation = robot.get_observation()
            current_pose = [observation[key] for key in ARX5_REAL_STATE_KEYS]
            actions = _predict_action_chunk(
                robot_observation=observation,
                dataset_features=dataset_features,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                task=args.task,
                robot_type=robot.robot_type,
                execution_horizon=execution_horizon,
                use_amp=policy_cfg.use_amp,
            )
            if args.safe_mode:
                actions = _clip_safe_actions(actions, current_pose=current_pose, max_joint_step=args.max_joint_step)
            if not actions:
                logger.warning("Policy returned no actions. Retrying.")
                time.sleep(0.1)
                continue

            grippers = [round(action[ARX5_REAL_STATE_KEYS[-1]], 4) for action in actions]
            logger.info("Predicted %d actions. Gripper values: %s", len(actions), grippers)
            _log_predicted_actions(actions)
            request_next_chunk = False

            if args.record_dir is not None:
                round_dir = _save_chunk_io(
                    record_dir=args.record_dir,
                    round_index=round_index,
                    observation=observation,
                    current_joint_state=robot.get_joint_vector(),
                    actions=actions,
                    camera_names=camera_names,
                )
                logger.info("Saved chunk %d to %s", round_index, round_dir)
                round_index += 1
                if args.safe_mode:
                    logger.info("SAFE MODE: press [I] for the next chunk.")
                continue

            state, request_next_chunk, running = _execute_chunk(
                robot=robot,
                actions=actions,
                step_duration_s=args.duration,
                keyboard=keyboard,
                safe_mode=args.safe_mode,
            )
            if args.safe_mode and state == LoopState.RUNNING:
                logger.info("SAFE MODE: press [I] for the next chunk.")
    finally:
        if keyboard is not None:
            keyboard.restore()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()

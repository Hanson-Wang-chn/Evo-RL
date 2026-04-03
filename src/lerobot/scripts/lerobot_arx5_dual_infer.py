#!/usr/bin/env python

"""Run dual-arm LeRobot Pi0.5 inference on two ARX5 arms."""

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
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import auto_select_torch_device

from lerobot.robots.arx5_follower.arx5_client import ARX5ArmClient


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
logger = logging.getLogger(__name__)

DEFAULT_POLICY_PATH = "lerobot/pi05_bimanual"
DEFAULT_LEFT_CAN_PORT = "can0"
DEFAULT_RIGHT_CAN_PORT = "can1"
DEFAULT_STATS_PATH = files("lerobot.robots.arx5_follower").joinpath("pi05_arx5_default_stats.json")
DUAL_STATE_DIM = 14
STATE_KEYS = tuple(f"state.{index}" for index in range(DUAL_STATE_DIM))


def _visual_image_slot_names(policy_cfg: PreTrainedConfig) -> list[str]:
    """Names after `observation.images.` in policy input order (must match training)."""
    if not policy_cfg.input_features:
        raise ValueError("Policy input_features is empty; cannot infer camera slot names.")
    prefix = f"{OBS_STR}.images."
    slots: list[str] = []
    for key, feat in policy_cfg.input_features.items():
        if not key.startswith(prefix) or feat.type is not FeatureType.VISUAL:
            continue
        slots.append(key.removeprefix(prefix))
    if not slots:
        raise ValueError("Policy has no VISUAL observation.images.* inputs.")
    return slots


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
    image_slot_names: list[str],
    camera_specs: dict[str, str],
    use_usb_cams: bool,
    width: int,
    height: int,
    fps: int,
) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for name in image_slot_names:
        if name not in camera_specs:
            raise ValueError(
                f"Missing camera spec for slot '{name}'. Required slots (from policy): {image_slot_names}. "
                f"Pass e.g. --cameras {name}:<serial> for each."
            )
        source = camera_specs[name]
        rotation = 0
        if use_usb_cams:
            configs[name] = OpenCVCameraConfig(
                index_or_path=int(source),
                width=width,
                height=height,
                fps=fps,
                rotation=rotation,
            )
        else:
            configs[name] = RealSenseCameraConfig(
                serial_number_or_name=source,
                width=width,
                height=height,
                fps=fps,
                rotation=rotation,
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


def _build_dataset_features(camera_configs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observation_features: dict[str, type | tuple[int, int, int]] = {key: float for key in STATE_KEYS}
    for name, camera_config in camera_configs.items():
        observation_features[name] = (camera_config.height, camera_config.width, 3)
    action_features = {key: float for key in STATE_KEYS}
    return combine_feature_dicts(
        hw_to_dataset_features(observation_features, OBS_STR),
        hw_to_dataset_features(action_features, ACTION),
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
    actions: list[dict[str, float]] = []
    for index in range(horizon):
        row = action_chunk[index]
        actions.append({name: float(row[offset]) for offset, name in enumerate(action_names)})
    return actions


def _save_chunk_io(
    *,
    record_dir: Path,
    round_index: int,
    observation: dict[str, Any],
    state: list[float],
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

    (input_dir / "state.json").write_text(
        json.dumps({"state": state}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "actions.json").write_text(json.dumps(actions, indent=2) + "\n", encoding="utf-8")
    return round_dir


def _log_predicted_actions(actions: list[dict[str, float]]) -> None:
    for action_index, action in enumerate(actions):
        logger.debug("Action[%d]: %s", action_index, action)


def _log_keyboard_help(*, safe_mode: bool) -> None:
    message = (
        "Keyboard: [Space] stop | [H] home | [B] teach | [N] record pose | "
        "[M] goto pose | [G] open grippers | [R] resume | [Q] quit"
    )
    if safe_mode:
        message += " | [I] next chunk (while running)"
    logger.info(message)


def _set_both_grippers(left_arm: ARX5ArmClient, right_arm: ARX5ArmClient, gripper_value: float) -> None:
    """Keep current joint angles; set catch_pos (index 6) on both arms."""
    left_pose = list(left_arm.get_state())
    right_pose = list(right_arm.get_state())
    left_pose[6] = float(gripper_value)
    right_pose[6] = float(gripper_value)
    left_arm.send_joint(left_pose)
    right_arm.send_joint(right_pose)
    logger.info("Both grippers set to %.4f (6-DOF joints unchanged).", gripper_value)


def _run_keyboard_command(
    *,
    key: str | None,
    left_arm: ARX5ArmClient,
    right_arm: ARX5ArmClient,
    state: LoopState,
    safe_mode: bool,
    request_next_chunk: bool,
    gripper_open_value: float,
) -> tuple[LoopState, bool, bool]:
    running = True
    if key == " ":
        left_arm.hold_position()
        right_arm.hold_position()
        logger.warning("Emergency stop: holding current poses.")
        return LoopState.STOPPED, False, running
    if key == "q":
        logger.info("Quit requested.")
        return state, request_next_chunk, False
    if key == "g":
        _set_both_grippers(left_arm, right_arm, gripper_open_value)
        return state, request_next_chunk, running
    if state == LoopState.STOPPED:
        if key == "h":
            logger.info("Moving both ARX5 arms to the home pose.")
            left_arm.hold_position()
            right_arm.hold_position()
            time.sleep(0.1)
            left_arm.go_home()
            right_arm.go_home()
            time.sleep(2.0)
            left_arm.hold_position()
            right_arm.hold_position()
        elif key == "b":
            left_arm.enter_teach_mode()
            right_arm.enter_teach_mode()
            logger.info(
                "Teach mode enabled. Drag both arms to the desired pose, then press [N] to save the poses."
            )
            return LoopState.TEACHING, request_next_chunk, running
        elif key == "m":
            if left_arm.has_recorded_pose() and right_arm.has_recorded_pose():
                left_arm.move_to_recorded()
                right_arm.move_to_recorded()
                logger.info("Moved both arms to the recorded poses.")
            else:
                logger.warning("No recorded poses found. Use [B] then [N] first.")
        elif key == "r":
            left_arm.hold_position()
            right_arm.hold_position()
            time.sleep(0.1)
            if safe_mode:
                logger.info("SAFE MODE: policy armed. Press [I] for each action chunk.")
            else:
                logger.info("Resumed continuous policy control.")
            return LoopState.RUNNING, not safe_mode, running
    elif state == LoopState.TEACHING:
        if key == "n":
            left_arm.save_recorded_pose()
            right_arm.save_recorded_pose()
            left_arm.hold_position()
            right_arm.hold_position()
            logger.info("Recorded poses saved for both arms.")
            return LoopState.STOPPED, False, running
    elif state == LoopState.RUNNING and safe_mode and key == "i":
        logger.info("SAFE MODE: next chunk requested.")
        return state, True, running
    return state, request_next_chunk, running


def _execute_dual_chunk(
    *,
    left_arm: ARX5ArmClient,
    right_arm: ARX5ArmClient,
    actions: list[dict[str, float]],
    step_duration_s: float,
    keyboard: KeyboardListener | None,
    safe_mode: bool,
    gripper_open_value: float,
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
                left_arm=left_arm,
                right_arm=right_arm,
                state=state,
                safe_mode=safe_mode,
                request_next_chunk=request_next_chunk,
                gripper_open_value=gripper_open_value,
            )
            if state != LoopState.RUNNING or not running:
                break
        left_joint = [float(action[f"state.{index}"]) for index in range(7)]
        right_joint = [float(action[f"state.{7 + index}"]) for index in range(7)]
        left_thread = threading.Thread(target=left_arm.send_joint, args=(left_joint,))
        right_thread = threading.Thread(target=right_arm.send_joint, args=(right_joint,))
        left_thread.start()
        right_thread.start()
        left_thread.join()
        right_thread.join()
        precise_sleep(max(step_duration_s - (time.perf_counter() - step_start), 0.0))
    return state, request_next_chunk, running


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LeRobot pi05 dual-arm policy on two ARX5 arms.",
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

    parser.add_argument("--left-can-port", type=str, default=DEFAULT_LEFT_CAN_PORT)
    parser.add_argument("--right-can-port", type=str, default=DEFAULT_RIGHT_CAN_PORT)
    parser.add_argument("--arm-type", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--use-stub", action="store_true")

    parser.add_argument(
        "--cameras",
        nargs="+",
        default=[
            "base:254522071216",
            "left_wrist:150622073629",
            "right_wrist:409122272986",
        ],
        help=(
            "One name:serial_or_index per policy image slot (keys match checkpoint "
            "observation.images.*, e.g. base/left_wrist/right_wrist)."
        ),
    )
    parser.add_argument("--use-usb-cams", action="store_true")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--list-cameras", action="store_true")

    parser.add_argument(
        "--safe-mode",
        action="store_true",
        default=False,
        help="After [R], require [I] for each policy chunk (same idea as lerobot_arx5_infer).",
    )
    parser.add_argument("--no-keyboard", action="store_true")
    parser.add_argument(
        "--gripper-open",
        type=float,
        default=4.0,
        help="Catch position for both arms when pressing [G] (matches typical training scale ~3–5).",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="If set, run policy each round and save observation + actions here; do not send joint commands (no motion).",
    )
    parser.add_argument(
        "--save-run-dir",
        type=Path,
        default=None,
        help="If set, also save each chunk (images, state.json, actions.json) here while executing on the arms. Mutually exclusive with --record-dir.",
    )

    args = parser.parse_args()

    if args.list_cameras:
        _list_realsense_cameras()
        return
    if not args.task:
        parser.error("--task is required unless --list-cameras is used.")
    if args.execution_horizon is not None and args.execution_horizon <= 0:
        parser.error("--execution-horizon must be positive.")
    if args.safe_mode and args.no_keyboard:
        parser.error("--safe-mode requires keyboard control (omit --no-keyboard).")
    if args.record_dir is not None and args.save_run_dir is not None:
        parser.error("Use either --record-dir (no motion) or --save-run-dir (with motion), not both.")

    logger.info("Loading policy config from %s", args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    state_feature = None if policy_cfg.input_features is None else policy_cfg.input_features.get(f"{OBS_STR}.state")
    if state_feature is None or state_feature.shape[0] != DUAL_STATE_DIM:
        raise ValueError(
            f"Dual-arm runtime expects observation.state dim={DUAL_STATE_DIM}, "
            f"but checkpoint declares {getattr(state_feature, 'shape', None)}."
        )
    action_feature = None if policy_cfg.output_features is None else policy_cfg.output_features.get(ACTION)
    if action_feature is None or action_feature.shape[0] != DUAL_STATE_DIM:
        raise ValueError(
            f"Dual-arm runtime expects action dim={DUAL_STATE_DIM}, "
            f"but checkpoint declares {getattr(action_feature, 'shape', None)}."
        )

    image_slot_names = _visual_image_slot_names(policy_cfg)
    logger.info("Policy image slots (use these in --cameras): %s", image_slot_names)

    camera_specs = _parse_camera_specs(args.cameras)
    extra_specs = sorted(set(camera_specs) - set(image_slot_names))
    if extra_specs:
        logger.warning("Ignoring --cameras entries not in policy: %s", extra_specs)

    camera_configs = _make_camera_configs(
        image_slot_names=image_slot_names,
        camera_specs=camera_specs,
        use_usb_cams=args.use_usb_cams,
        width=args.cam_width,
        height=args.cam_height,
        fps=args.fps,
    )

    left_recorded_pose = Path("checkpoints/left_recorded_pose.json")
    right_recorded_pose = Path("checkpoints/right_recorded_pose.json")
    left_arm = ARX5ArmClient(
        can_port=args.left_can_port,
        arm_type=args.arm_type,
        use_stub=args.use_stub,
        recorded_pose_path=left_recorded_pose,
    )
    right_arm = ARX5ArmClient(
        can_port=args.right_can_port,
        arm_type=args.arm_type,
        use_stub=args.use_stub,
        recorded_pose_path=right_recorded_pose,
    )

    policy_cfg, policy, preprocessor, postprocessor, device = _load_policy_bundle(
        policy_path=args.policy_path,
        device_override=args.policy_device,
        stats_path=args.stats_path,
        policy_cfg=policy_cfg,
    )
    dataset_features = _build_dataset_features(camera_configs)
    camera_names = list(camera_configs)
    execution_horizon = args.execution_horizon or int(policy_cfg.n_action_steps)
    keyboard = None if args.no_keyboard else KeyboardListener()
    state = LoopState.RUNNING if keyboard is None else LoopState.STOPPED
    request_next_chunk = keyboard is None or not args.safe_mode
    round_index = 0
    running = True

    cameras = {
        name: RealSenseCamera(config) if not args.use_usb_cams else OpenCVCameraConfig(
            index_or_path=config.index_or_path,
            width=config.width,
            height=config.height,
            fps=config.fps,
            rotation=config.rotation,
        )
        for name, config in camera_configs.items()
    }
    try:
        logger.info("Connecting cameras.")
        for name, camera in cameras.items():
            logger.info("Connecting camera '%s' via %s", name, camera)
            camera.connect()
            logger.info("Camera '%s' connected.", name)

        logger.info(
            "Dual-arm runtime ready. Left CAN=%s Right CAN=%s Stub=%s RecordDir=%s SaveRunDir=%s",
            args.left_can_port,
            args.right_can_port,
            args.use_stub,
            args.record_dir,
            args.save_run_dir,
        )
        if keyboard is not None:
            logger.info("Keyboard control enabled.")
            _log_keyboard_help(safe_mode=args.safe_mode)
            if args.safe_mode:
                logger.info("SAFE MODE: stopped by default. Press [R] to arm policy, then [I] for each chunk.")
            else:
                logger.info("Stopped by default: press [R] for continuous policy control.")

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        while running:
            if keyboard is not None:
                key = keyboard.get_key()
                previous_state = state
                state, request_next_chunk, running = _run_keyboard_command(
                    key=key,
                    left_arm=left_arm,
                    right_arm=right_arm,
                    state=state,
                    safe_mode=args.safe_mode,
                    request_next_chunk=request_next_chunk,
                    gripper_open_value=args.gripper_open,
                )
                if not running:
                    break
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

            left_state = left_arm.get_state()
            right_state = right_arm.get_state()
            combined_state = list(left_state[:7]) + list(right_state[:7])
            if len(combined_state) != DUAL_STATE_DIM:
                raise RuntimeError(f"Expected {DUAL_STATE_DIM}-dim state, got {len(combined_state)}.")

            observation: dict[str, Any] = {}
            for index, key in enumerate(STATE_KEYS):
                observation[key] = float(combined_state[index])

            for name, camera in cameras.items():
                observation[name] = camera.async_read()

            actions = _predict_action_chunk(
                robot_observation=observation,
                dataset_features=dataset_features,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                task=args.task,
                robot_type="arx5_dual",
                execution_horizon=execution_horizon,
                use_amp=policy_cfg.use_amp,
            )
            if not actions:
                logger.warning("Policy returned no actions. Retrying.")
                time.sleep(0.1)
                continue

            logger.info("Predicted %d actions.", len(actions))
            _log_predicted_actions(actions)
            request_next_chunk = False

            if args.save_run_dir is not None:
                run_round_dir = _save_chunk_io(
                    record_dir=args.save_run_dir,
                    round_index=round_index,
                    observation=observation,
                    state=combined_state,
                    actions=actions,
                    camera_names=camera_names,
                )
                logger.info("Saved chunk %d to %s (--save-run-dir, executing).", round_index, run_round_dir)
                round_index += 1

            if args.record_dir is not None:
                round_dir = _save_chunk_io(
                    record_dir=args.record_dir,
                    round_index=round_index,
                    observation=observation,
                    state=combined_state,
                    actions=actions,
                    camera_names=camera_names,
                )
                logger.info(
                    "Saved chunk %d to %s (--record-dir: skipped send_joint).",
                    round_index,
                    round_dir,
                )
                round_index += 1
                if args.safe_mode:
                    logger.info("SAFE MODE: press [I] for the next chunk.")
                continue

            state, request_next_chunk, running = _execute_dual_chunk(
                left_arm=left_arm,
                right_arm=right_arm,
                actions=actions,
                step_duration_s=args.duration,
                keyboard=keyboard,
                safe_mode=args.safe_mode,
                gripper_open_value=args.gripper_open,
            )
            if args.safe_mode and state == LoopState.RUNNING and running:
                logger.info("SAFE MODE: press [I] for the next chunk.")

    finally:
        if keyboard is not None:
            keyboard.restore()
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()


if __name__ == "__main__":
    main()


from types import SimpleNamespace

import pytest

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.scripts.lerobot_arx5_infer import (
    _get_required_policy_camera_keys,
    _make_camera_configs,
    _resolve_local_camera_map,
)


def test_multi_cups_checkpoint_maps_front_and_wrist_without_side():
    policy_cfg = SimpleNamespace(
        input_features={
            "observation.images.base": object(),
            "observation.images.right_wrist": object(),
            "observation.images.empty_camera_0": object(),
        }
    )

    required_policy_camera_keys = _get_required_policy_camera_keys(policy_cfg)
    local_camera_map = _resolve_local_camera_map(policy_cfg)
    configs = _make_camera_configs(
        camera_specs={"front": "0", "wrist": "1"},
        local_camera_map=local_camera_map,
        required_policy_camera_keys=required_policy_camera_keys,
        use_usb_cams=True,
        width=640,
        height=480,
        fps=30,
        flipped_local_cameras=set(),
    )

    assert required_policy_camera_keys == ["base", "right_wrist"]
    assert local_camera_map == {"base": "front", "right_wrist": "wrist"}
    assert set(configs) == {"base", "right_wrist"}
    assert all(isinstance(config, OpenCVCameraConfig) for config in configs.values())


def test_default_pi05_layout_still_requires_side_wrist_and_front():
    policy_cfg = SimpleNamespace(
        input_features={
            "observation.images.base_0_rgb": object(),
            "observation.images.left_wrist_0_rgb": object(),
            "observation.images.right_wrist_0_rgb": object(),
        }
    )

    required_policy_camera_keys = _get_required_policy_camera_keys(policy_cfg)
    local_camera_map = _resolve_local_camera_map(policy_cfg)

    assert required_policy_camera_keys == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    assert local_camera_map == {
        "base_0_rgb": "side",
        "left_wrist_0_rgb": "wrist",
        "right_wrist_0_rgb": "front",
    }


def test_multi_cups_checkpoint_raises_when_front_camera_missing():
    policy_cfg = SimpleNamespace(
        input_features={
            "observation.images.base": object(),
            "observation.images.right_wrist": object(),
            "observation.images.empty_camera_0": object(),
        }
    )

    with pytest.raises(ValueError, match="Missing required cameras"):
        _make_camera_configs(
            camera_specs={"wrist": "1"},
            local_camera_map=_resolve_local_camera_map(policy_cfg),
            required_policy_camera_keys=_get_required_policy_camera_keys(policy_cfg),
            use_usb_cams=True,
            width=640,
            height=480,
            fps=30,
            flipped_local_cameras=set(),
        )

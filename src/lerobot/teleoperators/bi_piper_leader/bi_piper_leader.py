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

import logging
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators.piper_leader import (
    PiperLeader,
    PiperLeaderConfig,
    PiperXLeader,
    PiperXLeaderConfig,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_bi_piper_leader import BiPiperLeaderConfig, BiPiperXLeaderConfig

logger = logging.getLogger(__name__)


class BiPiperLeader(Teleoperator):
    """Bimanual PiPER/PiPER-X leader arms."""

    config_class = BiPiperLeaderConfig
    name = "bi_piper_leader"
    _side_field_names = (
        "port",
        "judge_flag",
        "can_auto_init",
        "log_level",
        "startup_sleep_s",
        "manual_control",
        "prefer_ctrl_messages",
        "fallback_to_feedback",
        "sync_gripper",
        "gripper_effort_default",
        "gripper_status_code",
        "command_speed_ratio",
        "command_high_follow",
        "mode_refresh_interval_s",
        "enable_timeout_s",
        "gravity_comp_control_hz",
        "gravity_comp_tx_ratio",
        "gravity_comp_torque_limit",
        "gravity_comp_mit_kp",
        "gravity_comp_mit_kd",
        "gravity_comp_base_rpy_deg",
        "calibration_scale",
        "require_calibration",
        "disable_on_disconnect",
    )

    def _build_arm_config(self, arm_config_cls, side_cfg, side: str):
        kwargs = {name: getattr(side_cfg, name) for name in self._side_field_names}
        kwargs["id"] = f"{self.config.id}_{side}" if self.config.id else None
        kwargs["calibration_dir"] = self.config.calibration_dir
        return arm_config_cls(**kwargs)

    def __init__(self, config: BiPiperLeaderConfig | BiPiperXLeaderConfig):
        super().__init__(config)
        self.config = config

        if config.type == "bi_piperx_leader":
            arm_config_cls = PiperXLeaderConfig
            arm_cls = PiperXLeader
        else:
            arm_config_cls = PiperLeaderConfig
            arm_cls = PiperLeader

        left_arm_config = self._build_arm_config(arm_config_cls, config.left_arm_config, "left")
        right_arm_config = self._build_arm_config(arm_config_cls, config.right_arm_config, "right")

        self.left_arm = arm_cls(left_arm_config)
        self.right_arm = arm_cls(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_arm_features = self.left_arm.action_features
        right_arm_features = self.right_arm.action_features
        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def set_manual_control(self, enabled: bool) -> None:
        self.left_arm.set_manual_control(enabled)
        self.right_arm.set_manual_control(enabled)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action_dict: RobotAction = {}
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})
        return action_dict

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        left_feedback: dict[str, Any] = {}
        right_feedback: dict[str, Any] = {}
        for key, value in feedback.items():
            if key.startswith("left_"):
                left_feedback[key.removeprefix("left_")] = value
            elif key.startswith("right_"):
                right_feedback[key.removeprefix("right_")] = value
        self.left_arm.send_feedback(left_feedback)
        self.right_arm.send_feedback(right_feedback)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()


class BiPiperXLeader(BiPiperLeader):
    config_class = BiPiperXLeaderConfig
    name = "bi_piperx_leader"

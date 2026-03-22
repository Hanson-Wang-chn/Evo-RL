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

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class ARX5FollowerConfigBase:
    """Configuration for an ARX5 follower arm controlled through the vendor SDK."""

    port: str = "can0"
    arm_type: int = 0
    use_stub: bool = False
    startup_sleep_s: float = 0.2

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    gripper_min: float = 0.0
    gripper_max: float = 5.0
    max_translation_step_m: float | None = None

    protect_on_disconnect: bool = True
    recorded_pose_path: Path | None = None


@RobotConfig.register_subclass("arx5_follower")
@dataclass
class ARX5FollowerConfig(RobotConfig, ARX5FollowerConfigBase):
    def __post_init__(self):
        super().__post_init__()
        if self.arm_type not in {0, 1, 2}:
            raise ValueError("`arm_type` must be one of {0, 1, 2}.")
        if self.startup_sleep_s < 0:
            raise ValueError("`startup_sleep_s` must be >= 0.")
        if self.gripper_max <= self.gripper_min:
            raise ValueError("`gripper_max` must be greater than `gripper_min`.")
        if self.max_translation_step_m is not None and self.max_translation_step_m < 0:
            raise ValueError("`max_translation_step_m` must be >= 0 when provided.")

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

from pathlib import Path

from lerobot.robots.arx5_follower import ARX5_REAL_STATE_KEYS, ARX5_STATE_KEYS, ARX5Follower, ARX5FollowerConfig


def test_connect_observe_and_send_action(tmp_path: Path):
    robot = ARX5Follower(
        ARX5FollowerConfig(
            port="can0",
            use_stub=True,
            calibration_dir=tmp_path / "calibration",
            recorded_pose_path=tmp_path / "recorded_pose.json",
        )
    )
    robot.connect()
    try:
        assert robot.is_connected

        observation = robot.get_observation()
        assert set(observation) == set(ARX5_STATE_KEYS)
        assert all(observation[key] == 0.0 for key in ARX5_STATE_KEYS)

        action = {key: float(index + 1) for index, key in enumerate(ARX5_REAL_STATE_KEYS)}
        action.update({key: 99.0 for key in ARX5_STATE_KEYS if key not in action})
        sent_action = robot.send_action(action)

        for index, key in enumerate(ARX5_REAL_STATE_KEYS):
            expected = float(index + 1)
            if key == "gripper.pos":
                expected = min(expected, robot.config.gripper_max)
            assert sent_action[key] == expected
        for key in ARX5_STATE_KEYS:
            if key not in ARX5_REAL_STATE_KEYS:
                assert sent_action[key] == 0.0
    finally:
        robot.disconnect()


def test_recorded_pose_round_trip(tmp_path: Path):
    robot = ARX5Follower(
        ARX5FollowerConfig(
            port="can0",
            use_stub=True,
            calibration_dir=tmp_path / "calibration",
            recorded_pose_path=tmp_path / "recorded_pose.json",
        )
    )
    robot.connect()
    try:
        robot.send_action(
            {
                "ee.x": 0.12,
                "ee.y": 0.02,
                "ee.z": 0.18,
                "ee.roll": -0.1,
                "ee.pitch": 0.4,
                "ee.yaw": 0.05,
                "gripper.pos": 2.5,
            }
        )
        robot.exit_teach_mode_and_record()
        assert robot.has_recorded_pose()

        robot.send_action(
            {
                "ee.x": 0.2,
                "ee.y": 0.0,
                "ee.z": 0.1,
                "ee.roll": 0.0,
                "ee.pitch": 0.0,
                "ee.yaw": 0.0,
                "gripper.pos": 0.0,
            }
        )
        restored_action = robot.move_to_recorded(num_steps=1, step_interval_s=0.0)
        assert restored_action["ee.x"] == 0.12
        assert restored_action["gripper.pos"] == 2.5
    finally:
        robot.disconnect()

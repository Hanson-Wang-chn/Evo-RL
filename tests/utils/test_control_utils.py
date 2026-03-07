from types import SimpleNamespace

import pytest

from lerobot.utils.control_utils import sanity_check_bimanual_piper_pair


@pytest.mark.parametrize(
    ("robot_type", "teleop_type"),
    [
        ("bi_piper_follower", "bi_piper_leader"),
        ("bi_piperx_follower", "bi_piperx_leader"),
        ("so101_follower", "so101_leader"),
    ],
)
def test_sanity_check_bimanual_piper_pair_accepts_valid_pairs(robot_type, teleop_type):
    sanity_check_bimanual_piper_pair(
        SimpleNamespace(type=robot_type),
        SimpleNamespace(type=teleop_type),
    )


def test_sanity_check_bimanual_piper_pair_accepts_missing_teleop():
    sanity_check_bimanual_piper_pair(SimpleNamespace(type="bi_piperx_follower"), None)


@pytest.mark.parametrize(
    ("robot_type", "teleop_type"),
    [
        ("bi_piper_follower", "bi_piperx_leader"),
        ("bi_piperx_follower", "bi_piper_leader"),
        ("so101_follower", "bi_piperx_leader"),
        ("so101_follower", "bi_piper_leader"),
    ],
)
def test_sanity_check_bimanual_piper_pair_rejects_mixed_pairs(robot_type, teleop_type):
    with pytest.raises(ValueError, match="must be paired"):
        sanity_check_bimanual_piper_pair(
            SimpleNamespace(type=robot_type),
            SimpleNamespace(type=teleop_type),
        )

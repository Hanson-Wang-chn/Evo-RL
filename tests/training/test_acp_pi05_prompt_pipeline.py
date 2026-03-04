#!/usr/bin/env python

from unittest.mock import patch

import torch

from lerobot.configs.train import ACPConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.rl.acp_hook import build_acp_raw_batch_hook
from lerobot.rl.acp_tags import ACP_NEGATIVE_TAG, ACP_POSITIVE_TAG
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from tests.utils import require_package


class TrackingTokenizer:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(
        self,
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        padding_side="right",
        return_tensors="pt",
        **kwargs,
    ):
        del truncation, padding, padding_side, return_tensors, kwargs
        texts = [text] if isinstance(text, str) else list(text)
        self.calls.append(texts)

        batch_size = len(texts)
        input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
        for i, t in enumerate(texts):
            seq_len = min(max_length, max(1, len(t.split())))
            input_ids[i, :seq_len] = torch.arange(1, seq_len + 1, dtype=torch.long)
            attention_mask[i, :seq_len] = 1

        if isinstance(text, str):
            return {
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
            }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def _make_pi05_preprocessor():
    cfg = PI05Config(
        max_state_dim=4,
        max_action_dim=2,
        dtype="float32",
        device="cpu",
    )
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    preprocessor, _ = make_pi05_pre_post_processors(config=cfg, dataset_stats=None)
    return preprocessor


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_acp_prompt_reaches_pi05_tokenizer(mock_auto_tokenizer):
    tracking_tokenizer = TrackingTokenizer()
    mock_auto_tokenizer.from_pretrained.return_value = tracking_tokenizer

    preprocessor = _make_pi05_preprocessor()
    hook = build_acp_raw_batch_hook(
        ACPConfig(
            enable=True,
            indicator_field="complementary_info.acp_indicator",
            indicator_dropout_prob=0.0,
        ),
        seed=123,
    )
    batch = {
        "observation.state": torch.tensor(
            [
                [0.1, -0.2, 0.0, 0.3],
                [-0.4, 0.5, -0.6, 0.7],
            ],
            dtype=torch.float32,
        ),
        "task": ["Pick bottle", "Place bottle"],
        "complementary_info.acp_indicator": torch.tensor([1, 0], dtype=torch.int64),
    }

    conditioned_batch = hook(batch, 0)
    out = preprocessor(conditioned_batch)

    prompts = tracking_tokenizer.calls[-1]
    assert len(prompts) == 2
    assert ACP_POSITIVE_TAG in prompts[0]
    assert ACP_NEGATIVE_TAG in prompts[1]
    assert prompts[0].startswith("Task: Pick bottle")
    assert "State:" in prompts[0]
    assert OBS_LANGUAGE_TOKENS in out
    assert OBS_LANGUAGE_ATTENTION_MASK in out
    assert out[OBS_LANGUAGE_TOKENS].shape[0] == 2


@require_package("transformers")
@patch("lerobot.processor.tokenizer_processor.AutoTokenizer")
def test_without_acp_no_advantage_tag_in_pi05_tokenizer(mock_auto_tokenizer):
    tracking_tokenizer = TrackingTokenizer()
    mock_auto_tokenizer.from_pretrained.return_value = tracking_tokenizer

    preprocessor = _make_pi05_preprocessor()
    batch = {
        "observation.state": torch.tensor(
            [[0.0, 0.1, -0.1, 0.2]],
            dtype=torch.float32,
        ),
        "task": ["Pick bottle"],
    }
    _ = preprocessor(batch)

    prompts = tracking_tokenizer.calls[-1]
    assert len(prompts) == 1
    assert "Advantage:" not in prompts[0]
    assert prompts[0].startswith("Task: Pick bottle")

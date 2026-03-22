#!/usr/bin/env python

from types import SimpleNamespace

import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def test_fix_pytorch_state_dict_keys_maps_paligemma_lm_head_to_embed_tokens():
    policy = PI05Policy.__new__(PI05Policy)
    policy.model = SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(
            gemma_expert=SimpleNamespace(config=SimpleNamespace(use_adarms=False))
        )
    )

    state_dict = {
        "paligemma_with_expert.paligemma.lm_head.weight": torch.randn(4, 4),
    }

    fixed_state_dict = policy._fix_pytorch_state_dict_keys(state_dict, model_config=None)

    assert "paligemma_with_expert.paligemma.lm_head.weight" in fixed_state_dict
    assert "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight" in fixed_state_dict

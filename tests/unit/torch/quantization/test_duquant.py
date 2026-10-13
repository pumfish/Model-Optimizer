# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from modelopt.torch.quantization.duquant import (
    _base_rotation,
    build_duquant_transform,
    fold_duquant_transform,
)


def _official_base_rotation(block_size: int) -> torch.Tensor:
    """Independent reference for DuQuant's get_rot.py construction."""
    rotation = torch.eye(block_size, dtype=torch.float64)
    for index in range(block_size - 1):
        cosine = 1.0 / (index + 2) ** 0.5
        sine = (index + 1) ** 0.5 / (index + 2) ** 0.5
        plane = torch.eye(block_size, dtype=torch.float64)
        plane[index, index] = cosine
        plane[index + 1, index + 1] = cosine
        plane[index, index + 1] = -sine
        plane[index + 1, index] = sine
        rotation = plane @ rotation

    permutation = torch.arange(block_size)
    permutation[0], permutation[-1] = permutation[-1].item(), permutation[0].item()
    return rotation[permutation][:, permutation]


def test_duquant_base_rotation_matches_official_get_rot():
    """DuQuant's base rotation must match the official deterministic construction."""
    assert torch.equal(_base_rotation(8), _official_base_rotation(8))


def test_duquant_transform_preserves_linear_output():
    torch.manual_seed(0)
    linear = nn.Linear(128, 32, bias=True).float()
    reference = nn.Linear(128, 32, bias=True).float()
    reference.load_state_dict(linear.state_dict())

    transform, _ = build_duquant_transform(
        linear.weight, block_size=32, max_rotation_steps=4, permutation_times=1, seed=7
    )
    fold_duquant_transform(linear, transform)
    inputs = torch.randn(2, 3, 128)

    assert torch.allclose(
        linear(transform(inputs)), reference(inputs), atol=2e-5, rtol=2e-5
    )


def test_duquant_transform_is_deterministic():
    weight = torch.randn(32, 128)
    first_transform, first_weight = build_duquant_transform(
        weight, block_size=32, max_rotation_steps=3, permutation_times=1, seed=9
    )
    second_transform, second_weight = build_duquant_transform(
        weight, block_size=32, max_rotation_steps=3, permutation_times=1, seed=9
    )

    assert torch.equal(first_weight, second_weight)
    for index in range(first_transform.rotation_count):
        assert torch.equal(
            getattr(first_transform, f"rotation_{index}"),
            getattr(second_transform, f"rotation_{index}"),
        )


def test_duquant_transform_uses_official_seed_independent_rotation():
    """The official DuQuant rotation is deterministic and does not consume a random seed."""
    weight = torch.randn(32, 128)
    first_transform, first_weight = build_duquant_transform(
        weight, block_size=32, max_rotation_steps=3, permutation_times=1, seed=0
    )
    second_transform, second_weight = build_duquant_transform(
        weight, block_size=32, max_rotation_steps=3, permutation_times=1, seed=99
    )

    assert torch.equal(first_weight, second_weight)
    for index in range(first_transform.rotation_count):
        assert torch.equal(
            getattr(first_transform, f"rotation_{index}"),
            getattr(second_transform, f"rotation_{index}"),
        )

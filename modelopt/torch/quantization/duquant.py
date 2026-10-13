# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic rotation and permutation transforms used by DuQuant."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["DuQuantInputTransform", "build_duquant_transform", "fold_duquant_transform"]


def _base_rotation(block_size: int) -> torch.Tensor:
    """Build DuQuant's deterministic Givens base rotation.

    This matches ``get_rot.py`` in Hsu1023/DuQuant (MIT, commit
    d56cfc6fe97c34c0eb100fec82fe439865905679). The official transform does
    not add a random orthogonal matrix after the Givens construction.
    """
    if block_size <= 1 or block_size & (block_size - 1):
        raise ValueError(f"DuQuant block_size must be a power of two greater than one, got {block_size}")

    rotation = torch.eye(block_size, dtype=torch.float64)
    for index in range(block_size - 1):
        cosine = 1.0 / math.sqrt(index + 2)
        sine = math.sqrt(index + 1) / math.sqrt(index + 2)
        plane_rotation = torch.eye(block_size, dtype=torch.float64)
        plane_rotation[index, index] = cosine
        plane_rotation[index + 1, index + 1] = cosine
        plane_rotation[index, index + 1] = -sine
        plane_rotation[index + 1, index] = sine
        rotation = plane_rotation @ rotation

    exchange = torch.arange(block_size)
    exchange[0], exchange[-1] = exchange[-1].item(), exchange[0].item()
    rotation = rotation[exchange][:, exchange]

    return rotation


def _exchange_rotation(base_rotation: torch.Tensor, index: int) -> torch.Tensor:
    permutation = torch.arange(base_rotation.shape[0], device=base_rotation.device)
    permutation[0], permutation[index] = permutation[index].item(), permutation[0].item()
    return base_rotation[permutation][:, permutation]


def _apply_block_rotation(values: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    original_shape = values.shape
    block_size = rotation.shape[0]
    if original_shape[-1] % block_size:
        raise ValueError(
            f"DuQuant transform width {original_shape[-1]} is not divisible by block size {block_size}"
        )
    blocks = values.reshape(-1, block_size)
    return (blocks @ rotation.to(device=values.device, dtype=values.dtype)).reshape(original_shape)


def _search_rotation(weight: torch.Tensor, block_size: int, max_steps: int, _seed: int) -> torch.Tensor:
    """Run DuQuant's official greedy exchange search over the input-channel axis."""
    if max_steps <= 0:
        return torch.eye(block_size, device=weight.device, dtype=torch.float64)

    base_rotation = _base_rotation(block_size).to(weight.device)
    working = weight.detach().to(torch.float64).reshape(-1, block_size).clone()
    exchange_ids: list[int] = []
    peak_values: list[float] = []
    for _ in range(max_steps):
        maximum_index = working.argmax().item()
        minimum_index = working.argmin().item()
        max_row, max_column = divmod(maximum_index, block_size)
        min_row, min_column = divmod(minimum_index, block_size)
        peak_values.append(
            float((working[max_row, max_column] - working[min_row, min_column]).item())
        )
        exchange_id = (
            max_row
            if abs(working[max_row, max_column]) > abs(working[min_row, min_column])
            else min_row
        )
        exchange_ids.append(exchange_id)
        working = working @ _exchange_rotation(base_rotation, exchange_id % block_size)

    maximum_index = working.argmax().item()
    minimum_index = working.argmin().item()
    max_row, max_column = divmod(maximum_index, block_size)
    min_row, min_column = divmod(minimum_index, block_size)
    peak_values.append(
        float((working[max_row, max_column] - working[min_row, min_column]).item())
    )
    best_length = int(torch.tensor(peak_values).argmin().item())
    total_rotation = torch.eye(block_size, device=weight.device, dtype=torch.float64)
    for exchange_id in exchange_ids[:best_length]:
        total_rotation = total_rotation @ _exchange_rotation(base_rotation, exchange_id % block_size)
    return total_rotation


def _zigzag_permutation(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    """Balance per-block channel maxima with DuQuant's zigzag ordering."""
    width = weight.shape[-1]
    if width % block_size:
        raise ValueError(f"DuQuant width {width} is not divisible by block size {block_size}")
    group_count = width // block_size
    channel_max = weight.detach().abs().amax(dim=0)
    sorted_channels = torch.argsort(channel_max, descending=True).tolist()
    groups: list[list[int]] = [[] for _ in range(group_count)]
    group_index, increasing = 0, True
    for channel in sorted_channels:
        groups[group_index].append(channel)
        if increasing:
            group_index += 1
            if group_index == group_count:
                group_index -= 1
                increasing = False
        else:
            group_index -= 1
            if group_index < 0:
                group_index = 0
                increasing = True

    ordered = []
    for group in groups:
        ordered.extend(sorted(group, key=lambda index: float(channel_max[index]), reverse=True))
    return torch.tensor(ordered, device=weight.device, dtype=torch.long)


class DuQuantInputTransform(nn.Module):
    """A serializable rotation-permutation-rotation transform on the last axis."""

    def __init__(self, rotations: list[torch.Tensor], permutations: list[torch.Tensor]):
        super().__init__()
        if len(rotations) != len(permutations) + 1:
            raise ValueError("DuQuant requires one more rotation than permutations")
        self.rotation_count = len(rotations)
        self.permutation_count = len(permutations)
        for index, rotation in enumerate(rotations):
            self.register_buffer(f"rotation_{index}", rotation.detach().to(torch.float32))
        for index, permutation in enumerate(permutations):
            self.register_buffer(f"permutation_{index}", permutation.detach().to(torch.long))

    @property
    def block_size(self) -> int:
        return getattr(self, "rotation_0").shape[0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for index in range(self.permutation_count):
            outputs = _apply_block_rotation(outputs, getattr(self, f"rotation_{index}"))
            outputs = outputs.index_select(-1, getattr(self, f"permutation_{index}").to(outputs.device))
        return _apply_block_rotation(outputs, getattr(self, f"rotation_{self.rotation_count - 1}"))


@torch.no_grad()
def build_duquant_transform(
    weight: torch.Tensor,
    *,
    block_size: int = 128,
    max_rotation_steps: int = 256,
    permutation_times: int = 1,
    seed: int = 0,
) -> tuple[DuQuantInputTransform, torch.Tensor]:
    """Create a DuQuant transform and the equivalently folded Linear weight."""
    if weight.ndim != 2:
        raise ValueError(f"DuQuant only supports Linear weights, got shape {tuple(weight.shape)}")
    if permutation_times < 0:
        raise ValueError("permutation_times must be non-negative")
    if weight.shape[-1] % block_size:
        raise ValueError(
            f"DuQuant weight input width {weight.shape[-1]} is not divisible by block size {block_size}"
        )

    folded_weight = weight.detach().to(torch.float64).clone()
    rotations: list[torch.Tensor] = []
    permutations: list[torch.Tensor] = []
    for stage in range(permutation_times):
        rotation = _search_rotation(folded_weight, block_size, max_rotation_steps, seed + stage * 2)
        folded_weight = _apply_block_rotation(folded_weight, rotation)
        permutation = _zigzag_permutation(folded_weight, block_size)
        folded_weight = folded_weight.index_select(-1, permutation)
        rotations.append(rotation)
        permutations.append(permutation)

    final_rotation = _search_rotation(
        folded_weight, block_size, max_rotation_steps, seed + permutation_times * 2
    )
    folded_weight = _apply_block_rotation(folded_weight, final_rotation)
    rotations.append(final_rotation)
    transform = DuQuantInputTransform(rotations, permutations)
    return transform, folded_weight.to(device=weight.device, dtype=weight.dtype)


@torch.no_grad()
def fold_duquant_transform(linear: nn.Linear, transform: DuQuantInputTransform) -> None:
    """Fold ``transform`` into a Linear weight while preserving its FP output."""
    if linear.in_features % transform.block_size:
        raise ValueError("DuQuant transform block size is incompatible with Linear input width")
    transformed_weight = linear.weight.detach().to(torch.float64)
    for index in range(transform.permutation_count):
        transformed_weight = _apply_block_rotation(
            transformed_weight, getattr(transform, f"rotation_{index}")
        )
        transformed_weight = transformed_weight.index_select(
            -1, getattr(transform, f"permutation_{index}").to(transformed_weight.device)
        )
    transformed_weight = _apply_block_rotation(
        transformed_weight, getattr(transform, f"rotation_{transform.rotation_count - 1}")
    )
    linear.weight.copy_(transformed_weight.to(dtype=linear.weight.dtype))

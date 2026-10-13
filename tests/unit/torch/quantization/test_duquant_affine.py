# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DuQuant affine fake-quant backend."""

import torch

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.duquant_affine import (
    configure_duquant_lwc,
    finalize_duquant_lwc,
)
from modelopt.torch.quantization.nn import TensorQuantizer


def _reference_affine_quant(
    inputs: torch.Tensor, xmin: torch.Tensor, xmax: torch.Tensor, num_bits: int
) -> torch.Tensor:
    """Compute an independent affine Q/DQ reference."""
    qmax = (1 << num_bits) - 1
    scale = ((xmax - xmin) / qmax).clamp(min=1e-5, max=1e4)
    zero_point = (-xmin / scale).clamp(min=-1e4, max=1e4).round()
    quantized = (inputs / scale).round().add(zero_point).clamp(0, qmax)
    return (quantized - zero_point) * scale


def _quantizer(config: dict) -> TensorQuantizer:
    """Construct a ModelOpt quantizer with the DuQuant backend."""
    return TensorQuantizer(QuantizerAttributeConfig(**config))


def test_duquant_affine_backend_applies_swc_to_weight_ranges():
    """SWC must adjust the dynamic range used by a weight quantizer."""
    inputs = torch.tensor([[-8.0, -2.0, 3.0, 9.0], [-4.0, 0.0, 5.0, 12.0]])
    quantizer = _quantizer(
        {
            "num_bits": 8,
            "backend": "duquant_affine",
            "backend_extra_args": {
                "tensor_kind": "weight",
                "granularity": "per_channel_weight",
                "swc": 0.8,
            },
        }
    )

    xmin = inputs.amin(dim=-1, keepdim=True) * 0.8
    xmax = inputs.amax(dim=-1, keepdim=True) * 0.8
    assert torch.equal(quantizer(inputs), _reference_affine_quant(inputs, xmin, xmax, 8))


def test_duquant_affine_backend_applies_lac_to_per_token_activations():
    """LAC must adjust every activation token's dynamic range."""
    inputs = torch.tensor([[-8.0, -2.0, 3.0, 9.0], [-4.0, 0.0, 5.0, 12.0]])
    quantizer = _quantizer(
        {
            "num_bits": 8,
            "backend": "duquant_affine",
            "backend_extra_args": {
                "tensor_kind": "activation",
                "granularity": "per_token",
                "lac": 0.9,
            },
        }
    )

    xmin = inputs.amin(dim=-1, keepdim=True) * 0.9
    xmax = inputs.amax(dim=-1, keepdim=True) * 0.9
    assert torch.equal(quantizer(inputs), _reference_affine_quant(inputs, xmin, xmax, 8))


def test_duquant_affine_backend_persists_lwc_ranges():
    """LWC ranges must survive conversion from trainable logits to buffers."""
    weight = torch.tensor([[-8.0, -2.0, 3.0, 9.0, -4.0, 0.0, 5.0, 12.0]])
    quantizer = _quantizer(
        {
            "num_bits": 4,
            "backend": "duquant_affine",
            "backend_extra_args": {
                "tensor_kind": "weight",
                "granularity": "per_group_weight",
                "group_size": 4,
            },
        }
    )
    lower, upper = configure_duquant_lwc(quantizer, weight)
    with torch.no_grad():
        lower.copy_(torch.tensor([[[0.0], [1.0]]]))
        upper.copy_(torch.tensor([[[1.0], [0.0]]]))

    grouped = weight.reshape(1, 2, 4)
    xmin = grouped.amin(dim=-1, keepdim=True) * lower.sigmoid()
    xmax = grouped.amax(dim=-1, keepdim=True) * upper.sigmoid()
    expected = _reference_affine_quant(grouped, xmin, xmax, 4).reshape_as(weight)
    assert torch.equal(quantizer(weight), expected)

    finalize_duquant_lwc(quantizer, weight)
    assert "_duquant_lower_logits" not in quantizer._parameters
    assert "_duquant_upper_logits" not in quantizer._parameters
    assert torch.equal(quantizer(weight), expected)

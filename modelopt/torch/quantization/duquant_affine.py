# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Affine fake-quant backend matching DuQuant's clipping semantics."""

from __future__ import annotations

import torch
import torch.nn as nn

from .nn.modules.tensor_quantizer import TensorQuantizer, register_quant_backend

_BACKEND_NAME = "duquant_affine"
_LOWER_LOGITS = "_duquant_lower_logits"
_UPPER_LOGITS = "_duquant_upper_logits"
_XMIN = "_duquant_xmin"
_XMAX = "_duquant_xmax"
_CLIPMIN = 1e-5
_CLIPMAX = 1e4


def _range(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return inputs.detach().amin(dim=-1, keepdim=True), inputs.detach().amax(dim=-1, keepdim=True)


def _affine_fake_quant(
    inputs: torch.Tensor, xmin: torch.Tensor, xmax: torch.Tensor, num_bits: int
) -> torch.Tensor:
    qmax = (1 << num_bits) - 1
    scale = ((xmax - xmin) / qmax).clamp(min=_CLIPMIN, max=_CLIPMAX)
    zero_point = (-xmin / scale).clamp(min=-_CLIPMAX, max=_CLIPMAX).round()
    scaled = inputs / scale
    rounded = (scaled.round() - scaled).detach() + scaled
    quantized = (rounded + zero_point).clamp(0, qmax)
    return (quantized - zero_point) * scale


def _weight_inputs(quantizer: TensorQuantizer, weight: torch.Tensor) -> torch.Tensor:
    granularity = quantizer.backend_extra_args.get("granularity")
    if granularity == "per_channel_weight":
        return weight
    if granularity == "per_group_weight":
        group_size = quantizer.backend_extra_args.get("group_size")
        if not isinstance(group_size, int) or group_size <= 0:
            raise ValueError("per_group_weight requires a positive integer group_size")
        if weight.ndim != 2 or weight.shape[-1] % group_size:
            raise ValueError("DuQuant group-wise weights must be 2D and divisible by group_size")
        return weight.reshape(weight.shape[0], -1, group_size)
    raise ValueError(f"DuQuant weight granularity is unsupported: {granularity!r}")


def configure_duquant_lwc(
    quantizer: TensorQuantizer, weight: torch.Tensor
) -> tuple[nn.Parameter, nn.Parameter]:
    """Attach DuQuant's trainable two-sided LWC logits to a weight quantizer."""
    if quantizer.backend != _BACKEND_NAME:
        raise ValueError(f"LWC requires backend={_BACKEND_NAME}, got {quantizer.backend!r}")
    if quantizer.backend_extra_args.get("tensor_kind") != "weight":
        raise ValueError("DuQuant LWC applies only to weight quantizers")
    if quantizer.backend_extra_args.get("swc") is not None:
        raise ValueError("DuQuant LWC and SWC are mutually exclusive")
    if _LOWER_LOGITS in quantizer._parameters or _UPPER_LOGITS in quantizer._parameters:
        raise RuntimeError("DuQuant LWC parameters are already attached")
    inputs = _weight_inputs(quantizer, weight)
    shape = inputs.shape[:-1] + (1,)
    lower = nn.Parameter(torch.full(shape, 4.0, device=weight.device, dtype=torch.float32))
    upper = nn.Parameter(torch.full(shape, 4.0, device=weight.device, dtype=torch.float32))
    quantizer.register_parameter(_LOWER_LOGITS, lower)
    quantizer.register_parameter(_UPPER_LOGITS, upper)
    return lower, upper


def _lwc_range(
    inputs: torch.Tensor, lower_logits: torch.Tensor, upper_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xmin, xmax = _range(inputs)
    if lower_logits.shape != xmin.shape or upper_logits.shape != xmax.shape:
        raise ValueError(
            "DuQuant LWC shape mismatch: "
            f"ranges={tuple(xmin.shape)}, lower={tuple(lower_logits.shape)}, "
            f"upper={tuple(upper_logits.shape)}"
        )
    return xmin * lower_logits.sigmoid(), xmax * upper_logits.sigmoid()


@torch.no_grad()
def finalize_duquant_lwc(quantizer: TensorQuantizer, weight: torch.Tensor) -> None:
    """Replace temporary DuQuant LWC logits with checkpointed affine ranges."""
    lower = quantizer._parameters.pop(_LOWER_LOGITS, None)
    upper = quantizer._parameters.pop(_UPPER_LOGITS, None)
    if lower is None or upper is None:
        raise RuntimeError("DuQuant LWC parameters are not attached")
    inputs = _weight_inputs(quantizer, weight)
    xmin, xmax = _lwc_range(inputs, lower, upper)
    quantizer.register_buffer(_XMIN, xmin.to(device=weight.device, dtype=weight.dtype))
    quantizer.register_buffer(_XMAX, xmax.to(device=weight.device, dtype=weight.dtype))


def _ratio(extra_args: dict, key: str) -> float | None:
    value = extra_args.get(key)
    if value is None:
        return None
    if not isinstance(value, (float, int)) or not 0.0 < value <= 1.0:
        raise ValueError(f"DuQuant {key} must be in (0, 1], got {value!r}")
    return float(value)


def duquant_affine_backend(inputs: torch.Tensor, quantizer: TensorQuantizer) -> torch.Tensor:
    """Apply official DuQuant dynamic affine quantization and clipping."""
    if not isinstance(quantizer.num_bits, int) or not 2 <= quantizer.num_bits <= 16:
        raise ValueError(
            f"DuQuant affine requires integer bits in [2, 16], got {quantizer.num_bits!r}"
        )
    extra_args = quantizer.backend_extra_args
    tensor_kind = extra_args.get("tensor_kind")
    if tensor_kind == "weight":
        quant_inputs = _weight_inputs(quantizer, inputs)
        swc = _ratio(extra_args, "swc")
        if swc is not None:
            if _LOWER_LOGITS in quantizer._parameters or _UPPER_LOGITS in quantizer._parameters:
                raise RuntimeError("DuQuant LWC and SWC cannot both be enabled")
            xmin, xmax = _range(quant_inputs)
            xmin, xmax = xmin * swc, xmax * swc
        elif _LOWER_LOGITS in quantizer._parameters or _UPPER_LOGITS in quantizer._parameters:
            lower = quantizer._parameters.get(_LOWER_LOGITS)
            upper = quantizer._parameters.get(_UPPER_LOGITS)
            if lower is None or upper is None:
                raise RuntimeError("DuQuant LWC parameters unexpectedly became None")
            xmin, xmax = _lwc_range(quant_inputs, lower, upper)
        elif hasattr(quantizer, _XMIN) and hasattr(quantizer, _XMAX):
            xmin, xmax = getattr(quantizer, _XMIN), getattr(quantizer, _XMAX)
        else:
            xmin, xmax = _range(quant_inputs)
        return _affine_fake_quant(
            quant_inputs, xmin.to(quant_inputs), xmax.to(quant_inputs), quantizer.num_bits
        ).reshape_as(inputs)
    if tensor_kind == "activation":
        if extra_args.get("granularity") != "per_token":
            raise ValueError("DuQuant activation quantization requires per_token granularity")
        if _LOWER_LOGITS in quantizer._parameters or _UPPER_LOGITS in quantizer._parameters:
            raise ValueError("DuQuant activation quantizers do not support LWC")
        xmin, xmax = _range(inputs)
        lac = _ratio(extra_args, "lac")
        if lac is not None:
            xmin, xmax = xmin * lac, xmax * lac
        return _affine_fake_quant(inputs, xmin, xmax, quantizer.num_bits)
    raise ValueError(f"DuQuant tensor_kind must be 'weight' or 'activation', got {tensor_kind!r}")


def register_duquant_affine_backend() -> None:
    """Register the DuQuant backend during ModelOpt quantization import."""
    register_quant_backend(_BACKEND_NAME, duquant_affine_backend)


register_duquant_affine_backend()

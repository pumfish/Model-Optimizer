# LingBot-VA Quantization Reproduction

This `lingbotva-ptq` branch is a pinned Model-Optimizer dependency for the
LingBot-VA `control-aware-ptq` branch. It starts from upstream commit
`2fef374deda8ea0374ff89e54e566af79b683a5d` and adds the DuQuant components
needed to build or restore compatible checkpoints.

For the Chinese version, see
[lingbotva_quantization_reproduction_zh.md](lingbotva_quantization_reproduction_zh.md).

## Repository Responsibilities

| Method | Model-Optimizer responsibility | LingBot-VA responsibility |
| --- | --- | --- |
| BF16 | None | Reference inference |
| Uniform RTN W4A4 | Upstream ModelOpt quantization and checkpoint support | Checkpoint restore and inference |
| DuQuant W4A4 | Deterministic transform and affine Q/DQ backend | Wan topology, checkpoint restore, and inference |
| Omega-QVLA W4A4 | None beyond normal installation | Rotation pack loading and offline-scale Q/DQ |
| Ours W4A4 | None beyond normal installation | Online token-group Q/DQ and dual-branch mixed policy |

Omega-QVLA and Ours are deliberately implemented in the separate LingBot-VA
repository because they operate on LingBot-VA diffusion branches and their
external packs or policies. This repository provides only the ModelOpt-side
DuQuant extension.

## Installation

Use the official LingBot-VA server environment and install this exact branch:

```bash
git clone --branch lingbotva-ptq https://github.com/pumfish/Model-Optimizer.git
cd Model-Optimizer
pip install -e .
```

The paired LingBot-VA repository and its RoboTwin client retain their own
dependency setup. Follow its reproduction guide for Python, PyTorch, CUDA,
Diffusers, Transformers, and SAPIEN version requirements.

## Artifacts and Evaluation Boundary

Weights, checkpoints, Omega packs, mixed-precision policies, datasets, and
evaluation outputs are external artifacts. They are not included in this
repository. Obtain them through the corresponding LingBot-VA release workflow.

The LingBot-VA W4A4 experiments use simulated fake Q/DQ with floating-point
GEMM. They evaluate control accuracy; they do not measure integer-kernel
latency, throughput, or memory residency.

## Validation

Run the focused DuQuant tests in an environment where ModelOpt's optional CUDA
extensions can load:

```bash
python -m pytest -c /dev/null \
  tests/unit/torch/quantization/test_duquant.py \
  tests/unit/torch/quantization/test_duquant_affine.py
```

The LingBot-VA repository contains the end-to-end checkpoint restore and
RoboTwin evaluation smoke tests. This fork remains subject to the upstream
Model Optimizer license; LingBot-VA, RoboTwin, model, and dataset assets retain
their respective upstream licenses and terms.

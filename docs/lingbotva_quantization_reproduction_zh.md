# LingBot-VA 量化复现说明

`lingbotva-ptq` 分支是 LingBot-VA `control-aware-ptq` 分支使用的固定
Model-Optimizer 依赖。它以 NVIDIA 上游提交
`2fef374deda8ea0374ff89e54e566af79b683a5d` 为基线，仅增加构建或恢复兼容
DuQuant checkpoint 所需的组件。

英文版本见
[lingbotva_quantization_reproduction.md](lingbotva_quantization_reproduction.md)。

## 两个仓库的职责

| 方法 | Model-Optimizer 职责 | LingBot-VA 职责 |
| --- | --- | --- |
| BF16 | 无 | 参考推理 |
| Uniform RTN W4A4 | 上游 ModelOpt 的量化和 checkpoint 支持 | checkpoint 恢复与推理 |
| DuQuant W4A4 | 确定性变换与 affine Q/DQ backend | Wan 拓扑、checkpoint 恢复与推理 |
| Omega-QVLA W4A4 | 除正常安装外无额外职责 | rotation pack 加载与离线尺度 Q/DQ |
| Ours W4A4 | 除正常安装外无额外职责 | 在线 token-group Q/DQ 与双分支混合策略 |

Omega-QVLA 和 Ours 直接作用于 LingBot-VA 的扩散分支及其外部 pack/policy，
因此实现位于独立的 LingBot-VA 仓库；本仓库仅提供 ModelOpt 侧的 DuQuant 扩展。

## 安装

先按 LingBot-VA 官方说明配置 server 环境，再安装本固定分支：

```bash
git clone --branch lingbotva-ptq https://github.com/pumfish/Model-Optimizer.git
cd Model-Optimizer
pip install -e .
```

LingBot-VA 与 RoboTwin client 的 Python、PyTorch、CUDA、Diffusers、
Transformers 和 SAPIEN 版本要求以 LingBot-VA 复现文档为准。

## Artifact 与评测边界

模型权重、checkpoint、Omega pack、混合精度 policy、数据集和评测输出均是外部
artifact，不会提交到本仓库；请通过对应的 LingBot-VA 发布流程获取。

LingBot-VA 的 W4A4 实验使用浮点 GEMM 上的模拟 fake Q/DQ，评测目标是闭环控制
精度，不代表整数核延迟、吞吐量或显存驻留量。

## 验证

在可加载 ModelOpt 可选 CUDA 扩展的环境中运行 DuQuant 针对性测试：

```bash
python -m pytest -c /dev/null \
  tests/unit/torch/quantization/test_duquant.py \
  tests/unit/torch/quantization/test_duquant_affine.py
```

端到端 checkpoint 恢复和 RoboTwin 冒烟测试位于 LingBot-VA 仓库。本 fork 仍遵循
上游 Model Optimizer 许可证；LingBot-VA、RoboTwin、模型和数据集 asset 分别遵循
各自的上游许可证及使用条款。

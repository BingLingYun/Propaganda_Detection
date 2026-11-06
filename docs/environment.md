# 环境安装与设备验证

## 安装

以下为 NPU 训练环境示例。实际安装时需核对目标服务器的 Python、CPU 架构、CANN 和驱动版本；离线 wheel 的 Python ABI 也须匹配。

```bash
conda create -n propaganda-npu python=3.10 -y
conda activate propaganda-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

该示例对应的环境：

- 架构：`aarch64`
- Python：`3.10`
- CANN：`9.0.0`
- NPU：`910B4`

先安装 NPU 版 PyTorch：

```bash
python -m pip install -U pip setuptools wheel pyyaml
python -m pip install torch==2.7.1
python -m pip install torch-npu==2.7.1.post4
```

再安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

## 验证

```bash
python -c "import torch; import torch_npu; print(torch.__version__); print(torch_npu.__version__); print(torch.npu.is_available()); print(torch.npu.get_device_name(0))"
python -c "import torch, torch_npu; x=torch.randn(2,2).npu(); y=torch.randn(2,2).npu(); print(x @ y)"
```


CPU/CUDA 环境需先安装与本机匹配的 PyTorch，再安装 requirements.txt；只有 NPU 需要 torch-npu。所有命令从项目根目录运行。

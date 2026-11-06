# Propaganda Detection

宣传技巧片段检测。输入文本，输出字符级片段位置和一个或多个宣传技巧标签，可接入自己的数据集。

包含四类方法：`majority`、传统 `crf_svm`、神经网络 `mhbio_crf`，以及两阶段 `bilou_select`。仓库提供训练、预测、评估源码、配置说明与运行示例。

## 目录

```text
configs/             数据集、模型和训练配置，以及自定义数据集模板
src/
  common/            设备适配、配置/I/O 和实验辅助工具
  preprocess/        数据预处理
  baselines/         简单基线
  models/            多头 CRF、BILOU detector 和 span classifier
  modeling/          BIO-CRF 窗口数据与推理
  training/          共享配置解析与第二阶段训练数据
  inference/         共享 checkpoint 加载与两阶段推理
  evaluation/        统一片段评分与泛化指标
scripts/             训练、预测、阈值调优与数据准备入口
tests/               配置、checkpoint 与评分回归测试
docs/                配置说明、数据接口与运行示例
```

## 快速开始

Python 3.10 或更高版本。先按目标设备安装 PyTorch；NPU 还需匹配服务器 CANN、驱动和架构的 `torch-npu`。详细步骤见 [环境安装](docs/environment.md)。

```bash
python -m pip install -r requirements.txt
```

参照 [自定义数据集配置](configs/examples/custom_dataset.yaml)，填写自己的数据集名称、语言、数据路径和技巧标签，并准备 `data/processed/<dataset>/` 中的 JSONL 数据，具体格式见 [数据准备](docs/data.md)。

训练多头 BIO-CRF：

```bash
python scripts/train.py \
  --dataset_config configs/examples/custom_dataset.yaml \
  --model_config configs/models/xlm_mhbio_crf.yaml \
  --run_name mhbio_demo \
  --device npu --batch_size 4 --grad_accumulation 4 --epochs 10 --fp16
```

使用训练好的 checkpoint 预测：

```bash
python scripts/predict.py \
  --checkpoint outputs/custom/mhbio_demo/checkpoints/best \
  --input /path/to/input.jsonl \
  --out_dir outputs/custom/mhbio_demo/predictions \
  --split demo --device npu
```

输入每行包含 `id` 和 `text`；输出每行包含 `id` 和 `spans`。片段用 `start`、`end`、`technique` 表示，字符偏移从 0 开始，采用 Unicode 码点和左闭右开区间 `[start, end)`。

- [配置文件如何填写、参数覆盖顺序、数据接口](docs/configuration.md)
- [完整训练、预测、两阶段模型与评估示例](docs/usage.md)

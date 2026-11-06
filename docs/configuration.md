# 配置文件与数据接口

所有 Python 和 Bash 命令均从项目根目录执行。数据集、模型、训练参数分开配置：

| 文件 | 作用 |
| --- | --- |
| `configs/examples/custom_dataset.yaml` | 自定义数据集名称、语言、路径和技巧标签 |
| `configs/examples/custom_raw_dataset.yaml` | 原始数据字段映射、偏移约定和预处理配置 |
| `configs/models/xlm_mhbio_crf.yaml` | XLM-R 多头 BIO-CRF 模型及其推荐超参数 |
| `configs/models/monolingual_mhbio_crf.yaml` | 按 `language` 选择单语编码器 |
| `configs/models/xlm_bilou_select.yaml` | 两阶段模型的公共参数和 `stage1` / `stage2` 参数 |
| `configs/train_default.yaml` | 通用训练默认值 |

## 参数如何覆盖

优先级从低到高为：训练配置 → 模型配置 → 当前阶段的配置 → 命令行参数。例如模型配置中 `batch_size: 8` 会覆盖训练默认值中的 `16`，命令行 `--batch_size 2` 最终生效。

`--model_config`、`--train_config`、`--dataset_config` 分别选择这三类配置。训练结束后，实际生效参数保存在 `outputs/<dataset>/<run>/checkpoints/resolved_config.json`。

| 参数 | 含义 |
| --- | --- |
| `model_name` / `--model_name` | Hugging Face 模型名或本地编码器目录；离线环境使用本地目录 |
| `max_length` | 单个窗口的最大 token 数，须符合编码器长度限制 |
| `stride` | 相邻滑动窗口重叠的 token 数；小于可用窗口长度 |
| `batch_size` | 每步文档窗口或候选片段数量 |
| `grad_accumulation` / `grad_accum` | 梯度累积步数；有效 batch 约为两者乘积 |
| `encoder_lr` / `lr_encoder` | 编码器学习率 |
| `head_lr` / `lr_head` | BIO-CRF 分类头学习率 |
| `classifier_lr` | 两阶段第二阶段学习率 |
| `epochs`、`early_stopping_patience` | 最大训练轮数和 Dev 分数不再提升时的等待轮数 |
| `fp16` | 半精度训练；CPU 自动关闭，命令行可用 `--no_fp16` |
| `lazy_windows` | mhbio_crf 按需构造训练窗口，降低主机内存占用 |
| `candidate_chunk_size` | 两阶段推理每块最多处理的窗口或候选数量 |
| `max_steps`、`max_train_docs` | 限制训练步数或文档数，用于冒烟验证 |

设备由命令行 `--device auto|npu|cuda|cpu` 控制。当前训练为单进程单卡。

## 接入自己的数据

复制 `configs/examples/custom_dataset.yaml`，修改 `name`、`language`、`processed_dir` 和 `labels`。也可以用 `label_file: data/processed/custom/labels.json` 替代内嵌 `labels`；该文件结构为 `{"labels": ["technique_a", "technique_b"]}`。标签名称和顺序应在训练与推理间保持一致。

训练目录包含 `train.jsonl`、`dev.jsonl`；执行完整评估还需 `id_eval.jsonl` 和 `eval.jsonl`。每行一个 JSON 对象，例如以下人工示例：

```json
{"id":"example-1","text":"这只是一个示例。","spans":[{"start":0,"end":2,"technique":"technique_a"}]}
```

字符偏移按 Unicode 码点计数，从 0 开始，使用左闭右开区间 `[start, end)`；同一区间可以出现多个不同技巧标签。预测输入只需 `id` 与 `text`，不需要 `spans`。

```json
{"id":"example-2","text":"这是待预测的文本。"}
```

如需分组泛化评估，可提供 `eval_groups.json`，使用 `{ "文档ID": "目标域名" }` 或 `{ "文档ID": ["目标域1", "目标域2"] }`。数据准备流程见 [数据准备](data.md)。

## 选择两阶段模型

`xlm_bilou_select.yaml` 中 `stage1` 训练 BILOU 位置检测器，`stage2` 训练候选片段的多标签分类器。直接运行 `train.py` 时，pipeline 配置默认选择第一阶段；显式指定 `--model_type ts_classifier` 选择第二阶段，并通过 `--proposal_file` 提供训练候选。

第二阶段默认 `positive_overlap_mode: gold_coverage`：候选覆盖某个金标片段达到阈值时，将对应技巧设为正标签。阈值调优使用 Dev 集，命令示例见 [训练与预测](usage.md)。

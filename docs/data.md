# 准备自己的数据集

## 标准 JSONL 格式

在数据集配置中填写 `name`、`language`、`processed_dir` 和 `labels`。例如 `configs/examples/custom_dataset.yaml` 对应以下目录：

```text
data/processed/custom/
  train.jsonl
  dev.jsonl
  id_eval.jsonl
  eval.jsonl
  labels.json
  eval_groups.json
```

训练需要 `train.jsonl` 和 `dev.jsonl`。运行完整方法对比脚本还需要 `id_eval.jsonl` 和 `eval.jsonl`；单文件预测可以指定任意 JSONL 路径。`eval_groups.json` 仅在需要按域分组评估时使用。

每行代表一个文档，以下为人工格式示例：

```json
{"id":"example-1","text":"这只是一个示例。","spans":[{"start":0,"end":2,"technique":"technique_a"}]}
```

- `id`：字符串，文档 ID 在各划分之间唯一。
- `text`：原始文本；准备标签后不要再改变文本或字符偏移。
- `spans`：片段列表，无标注时为空列表。
- `start`、`end`：Unicode 码点偏移，从 0 开始，左闭右开 `[start, end)`。
- `technique`：技巧标签，必须与配置或 `labels.json` 中的字符串完全一致。同一片段可以输出多个不同技巧标签。

`labels.json` 的格式为：

```json
{"labels":["technique_a","technique_b"]}
```

训练可以使用配置中内嵌的 `labels`，或用 `label_file` 指向上述文件。评估时同目录的 `labels.json` 用于确定 Macro-F1 的完整标签集合。

## 预测输入

预测只需要文档 ID 和文本：

```json
{"id":"example-2","text":"这是待预测的文本。"}
```

输出包含对应的 `id` 和模型预测的 `spans`。

## 从原始文件预处理

如果自己的数据需要字段映射、偏移归一化或自动划分，复制 `configs/examples/custom_raw_dataset.yaml`，修改原始文件路径、字段名、偏移约定和标签后执行：

```bash
python -m src.preprocess.preprocess --config configs/examples/custom_raw_dataset.yaml
```

模板默认读取内嵌片段的 JSONL 文件；预处理器还支持 JSON、CSV、TSV，以及单独的标注文件。通过 `raw.docs_format`、`raw.spans_source` 和 `raw.field_map` 指定读取方式。

`raw.offset_convention` 控制字符/词偏移、索引起点和结束位置是否包含；输出统一为字符级左闭右开区间。该流程会从训练与验证文档池抽取 `id_eval_ratio` 比例的 ID Eval，并检查文档隔离、标签和偏移；不要在已经固定好四个划分时重复运行此步骤。

## 分组与导出

需要分组评估时，提供 `eval_groups.json`：

```json
{"example-2":"domain_a","example-3":["domain_a","domain_b"]}
```

将标准数据导出为 CSV：

```bash
python scripts/export_processed_csv.py --dataset_config configs/examples/custom_dataset.yaml
```

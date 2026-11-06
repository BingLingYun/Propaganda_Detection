# 训练、预测与评估示例

以下命令从项目根目录运行，使用自定义数据集配置。安装步骤见 [环境说明](environment.md)，数据格式和覆盖规则见 [配置说明](configuration.md)。

## 多头 BIO-CRF

使用自定义数据集模板，以明确的运行名保存结果：

```bash
python scripts/train.py \
  --dataset_config configs/examples/custom_dataset.yaml \
  --model_config configs/models/xlm_mhbio_crf.yaml \
  --run_name mhbio_demo \
  --device npu --batch_size 4 --grad_accumulation 4 --epochs 10 --fp16
```

CPU 冒烟验证时，将设备和训练规模改为 `--device cpu --batch_size 1 --epochs 1 --max_steps 1 --max_train_docs 2 --no_fp16`；离线环境追加 `--model_name /path/to/local/encoder`。主机内存紧张可追加 `--lazy_windows --lazy_window_cache_docs 2`。

```bash
python scripts/predict.py \
  --checkpoint outputs/custom/mhbio_demo/checkpoints/best \
  --input /path/to/input.jsonl \
  --out_dir outputs/custom/mhbio_demo/predictions \
  --split demo --device npu --batch_size 4
```

输出为 `outputs/custom/mhbio_demo/predictions/demo.jsonl`，每行包含 `id` 和 `spans`。预测时从 checkpoint 恢复配置、标签和分词器，不再传训练配置。

## 两阶段 BILOU + selected-span classifier

填写自己的数据集配置后，可以通过脚本完成四类方法的训练、阈值调优和评估：

```bash
DEVICE=npu MODEL_NAME=/path/to/local/encoder \
  bash scripts/run_dataset_baselines.sh configs/examples/custom_dataset.yaml
```

只运行两阶段模型时，可在命令前增加 `RUN_TRIVIAL=0 RUN_TRADITIONAL=0 RUN_MH=0`。脚本需要 `train`、`dev`、`id_eval`、`eval` 四个 split，Linux 或 WSL/Git Bash 环境可用。

单独控制两阶段训练的示例：

```bash
python scripts/train.py \
  --dataset_config configs/examples/custom_dataset.yaml \
  --model_config configs/models/xlm_bilou_select.yaml \
  --model_type ts_proposal --run_name proposal_demo --device npu

python scripts/predict_proposals.py \
  --checkpoint outputs/custom/proposal_demo/checkpoints/best \
  --input data/processed/custom/train.jsonl --split train \
  --out_dir outputs/custom/proposal_demo/proposals --device npu

python scripts/predict_proposals.py \
  --checkpoint outputs/custom/proposal_demo/checkpoints/best \
  --input data/processed/custom/dev.jsonl --split dev \
  --out_dir outputs/custom/proposal_demo/proposals --device npu

python -c "from pathlib import Path; p=Path('outputs/custom/proposal_demo/proposals'); (p/'train_dev.jsonl').write_bytes((p/'train.jsonl').read_bytes() + (p/'dev.jsonl').read_bytes())"

python scripts/train.py \
  --dataset_config configs/examples/custom_dataset.yaml \
  --model_config configs/models/xlm_bilou_select.yaml \
  --model_type ts_classifier --run_name classifier_demo \
  --proposal_file outputs/custom/proposal_demo/proposals/train_dev.jsonl --device npu

python scripts/predict_two_stage.py \
  --proposal_checkpoint outputs/custom/proposal_demo/checkpoints/best \
  --classifier_checkpoint outputs/custom/classifier_demo/checkpoints/best \
  --input /path/to/input.jsonl --split demo \
  --out_dir outputs/custom/two_stage_demo/predictions --device npu
```

第二阶段训练会使用候选文件中可匹配的训练/验证文档；合并文件仅供候选查找，训练与 Dev 文档仍按原 split 分离。自动脚本还会调优两阶段阈值。Stage 1 checkpoint 仅用于生成候选，端到端技巧预测使用 `predict_two_stage.py`。

单独调优第一阶段阈值：

```bash
python scripts/tune_thresholds.py \
  --checkpoint outputs/custom/proposal_demo/checkpoints/best \
  --gold data/processed/custom/dev.jsonl --device npu
```

第二阶段调优还需 `--proposal_file` 指向 Dev 候选文件。调优结果写入 checkpoint 的 `thresholds.json`，后续预测自动读取。

## 传统方法

```bash
bash scripts/run_traditional_crf_svm.sh configs/examples/custom_dataset.yaml
python -m src.baselines.trivial_baselines --help
```

`crf_svm` 使用二值 BIO CRF 定位，再由 TF-IDF + LinearSVC 预测技巧；`majority` 为简单对照。这两种方法使用 CPU。

## 评估

```bash
python -m src.evaluation.evaluate \
  --gold data/processed/custom/eval.jsonl \
  --pred outputs/custom/mhbio_demo/predictions/demo.jsonl \
  --out_json outputs/custom/mhbio_demo/reports/eval.json \
  --out_md outputs/custom/mhbio_demo/reports/eval.md
```

预测输入应与评估金标对应同一组文档 ID。当前评分版本为 `span-coverage-v2`：同类覆盖区间按并集计算部分匹配分，避免重叠字符重复计分。Gold 同目录的 `labels.json` 用作 Macro-F1 标签全集；自定义数据建议准备此文件。`--merge_same_label_overlaps` 会额外改变片段统计单元。

## 输出结构

```text
outputs/<dataset>/<run>/
  checkpoints/
    resolved_config.json
    runtime.json
    trainer_log.jsonl
    best/
      meta.json
      model.pt
      encoder/
      tokenizer/
  predictions/
  reports/
```

`best/` 为完整推理 checkpoint。

# Overview

This repository contains instructions for reproducing the fine-tuning and evaluation process of predicting the next purchase item using a generative language model (LLM).

*Prerequisites*
```bash
pip install -U transformers accelerate datasets peft trl bitsandbytes wandb
```

# Training

## Preparation

Uncompress raw_data.tar.gz at the repo's folder

```bash
tar -xvzf raw_dta.tar.gz
```

This will create a folder structure under ./raw_data and populate it with the necessary data files.

## Dataset Generation

Explanation based on concurrent purchase (You can do similarily for single purchase)

### Generate train dataset

Modify `args/run-rewardtrainer-katchers-concurrent-args.json` as follows:
- `build_dataset`: true
- `build_split`: "train"

Execute the following:
```bash
python src/rewardtrainer-llm-for-rec.py args/run-rewardtrainer-katchers-concurrent-args.json
```

The resulting dataset will be generated in the datasets/katchers/concurrent/train/ folder.

### Generate validation dataset

Modify `args/run-rewardtrainer-katchers-concurrent-args.json` as follows:
- `build_dataset`: true
- `build_split`: "val"

Execute the following:
```bash
python src/rewardtrainer-llm-for-rec.py args/run-rewardtrainer-katchers-concurrent-args.json
```

The resulting dataset will be generated in the datasets/katchers/concurrent/val/ folder.

## Training

Modify `args/run-rewardtrainer-katchers-concurrent-args.json` as follows
- `build_dataset`: false

Execute the following:
```bash
python src/rewardtrainer-llm-for-rec.py args/run-rewardtrainer-katchers-concurrent-args.json
```

Fine-tuned checkpoints will be generated in the output/rec-sys-llm-concurrent-tuning folder.

# Testing

## Dataset Generation

Explanation based on concurrent purchase (You can do similarily for single purchase)

### Generate test dataset

Modify `args/run-rewardevaluator-katchers-concurrent-args.json` as follows:
- `build_dataset`: true
- `calc_metric_only`: false

Execute the following:
```bash
python src/rewardevaluator-llm-for-rec.py args/run-rewardevaluator-katchers-concurrent-args.json
```

The resulting dataset will be generated in the datasets/katchers/concurrent/test/ folder.

## Evaluation

### Perform predictions and metric calculations on the test dataset

Modify `args/run-rewardevaluator-katchers-concurrent-args.json` as follows:
- `build_dataset`: false
- `calc_metric_only`: false

Execute the following:
```bash
python src/rewardevaluator-llm-for-rec.py args/run-rewardevaluator-katchers-concurrent-args.json
```

The prediction results will be saved in the ./evaluation_katchers_concurrent.pkl file.

### Calculate metrics only from the test dataset prediction results

Modify `args/run-rewardevaluator-katchers-concurrent-args.json` as follows:
- `build_dataset`: false
- `calc_metric_only`: true

Execute the following:
```bash
python src/rewardevaluator-llm-for-rec.py args/run-rewardevaluator-katchers-concurrent-args.json
```

NDCG@5 and NDCG@10 will be displayed on the console as the result.
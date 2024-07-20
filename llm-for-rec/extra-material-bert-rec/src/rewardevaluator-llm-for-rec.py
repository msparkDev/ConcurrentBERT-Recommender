import os
from dataclasses import dataclass, field
from typing import Optional
import re
import math
import pandas as pd
import numpy as np

import torch
import sys
import tyro
from accelerate import Accelerator
from datasets import load_dataset, Dataset, load_from_disk
from peft import AutoPeftModelForSequenceClassification, LoraConfig
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorWithPadding,
    logging as hf_logging,
)
import logging
from trl import RewardConfig, RewardTrainer, trainer


@dataclass
class ScriptArguments:
    cache_dir: Optional[str] = field(
        default="/Jupyter/huggingface/.cache", metadata={"help": "the cache dir"}
    )

    model_name_or_path: Optional[str] = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={"help": "the model name or path"},
    )

    raw_data_path: Optional[str] = field(
        default="/Jupyter/dev_src//Jupyter/dev_src/rec-sys-by-llm/data/katchers/concurrent",
        metadata={"help": "the raw data path"},
    )

    build_dataset: Optional[bool] = field(
        default=False, metadata={"help": "whether to build the dataset"}
    )

    calc_metric_only: Optional[bool] = field(
        default=False, metadata={"help": "is calculating metric only mode?"}
    )

    build_split: Optional[str] = field(
        default="test", metadata={"help": "the split to build the dataset"}
    )

    num_negatives: Optional[int] = field(
        default=49,
        metadata={"help": "the number of negative examples per positive example"},
    )

    dataset_path: Optional[str] = field(
        default="/Jupyter/dev_src//Jupyter/dev_src/rec-sys-by-llm/datasets/katchers/concurrent",
        metadata={"help": "the raw data path"},
    )

    num_workers: Optional[int] = field(
        default=8, metadata={"help": "the number of workers"}
    )

    model_type: Optional[str] = field(
        default="llama2",
        metadata={"help": "You should choose one of mistral, llama2, phi2, midm"},
    )

    packing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use packing for SFTTrainer"}
    )

    evaluation_path: Optional[str] = field(
        default="./evaluation.pkl",
        metadata={"help": "test evaluation raw result path"},
    )


def preprocess_function(examples, tokenizer, max_length):
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
    }

    for example in examples["text"]:
        tokenized_text = tokenizer(example, truncation=True, max_length=max_length)

        new_examples["input_ids"].append(tokenized_text["input_ids"])
        new_examples["attention_mask"].append(tokenized_text["attention_mask"])

    return new_examples


############
def forward_pass(batch, model=None, collator=None, device=None):
    n = 0
    for k, v in batch.items():
        n = len(v)
        break

    data = [{k: v[i] for k, v in batch.items()} for i in range(n)]

    aa = collator(data)
    # del aa["return_loss"]

    inputs = {k: v.to(device) for k, v in aa.items()}

    with torch.no_grad():
        prediciton = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

    return {"logit": prediciton.logits.cpu()}


def calculate_ndcg(group_data, k=10):
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG) for a group of predictions.

    Parameters:
    - group_data: A DataFrame with columns 'score' and 'label', sorted by 'score'.
    - k: The number of items to consider in the calculation.

    Returns:
    - The NDCG score for the group.
    """
    group_data = group_data.sort_values(by="logit", ascending=False)
    rel = group_data["label"].tolist()
    dcg = sum([rel[i] / math.log2(i + 2) for i in range(k)])
    idcg = sum(
        [
            sorted(rel, reverse=True)[i] / math.log2(i + 2)
            for i in range(min(k, len(rel)))
        ]
    )
    return dcg / idcg if idcg > 0 else 0


def evaluate():
    # model_args = tyro.cli(ScriptArguments)
    # logging.info(model_args)

    parser = HfArgumentParser((ScriptArguments, RewardConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    ndcg5_values, ndcg10_values = [], []

    if model_args.calc_metric_only:
        df = pd.read_pickle(model_args.evaluation_path)
        for key, group in df.groupby("id"):
            group_data = group.reset_index(drop=True)
            ndcg5_values.append(calculate_ndcg(group_data, 5))
            ndcg10_values.append(calculate_ndcg(group_data, 10))
            # print(group)
        print(
            f"Average NDCG@5: {np.mean(ndcg5_values)}\nAverage NDCG@10: {np.mean(ndcg10_values)}"
        )
        return

    if model_args.build_dataset:
        # 데이터셋 빌드만 하고 종료
        build_dataset_from_rawdataset(model_args, training_args)
        return

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )

    finetuned_model = AutoPeftModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        num_labels=1,
        quantization_config=bnb_config,
    )
    finetuned_model.use_cache = False
    finetuned_model.config.pad_token_id = tokenizer.pad_token_id
    finetuned_model.eval()

    # 토크나이저 설정은 되어 있다고 가정

    tokenize_function = lambda examples: preprocess_function(
        examples, tokenizer=tokenizer, max_length=training_args.max_length
    )

    eval_dataset = load_from_disk(os.path.join(model_args.dataset_path, "test"))

    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )

    eval_dataset = eval_dataset.select_columns(
        ["input_ids", "attention_mask", "label", "id"]
    )
    eval_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    # eval_dataset = eval_dataset.select(range(32))

    # collator = trainer.utils.RewardDataCollatorWithPadding(
    #     tokenizer=tokenizer, max_length=training_args.max_length
    # )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer, max_length=training_args.max_length
    )

    logging.info(f"Start Evaluation... with {model_args.model_type}")
    result_ds = eval_dataset.map(
        lambda x: forward_pass(
            x, model=finetuned_model, collator=collator, device="cuda"
        ),
        batched=True,
        batch_size=8,
    )
    result_ds.set_format("pandas")
    result_df = result_ds.data.to_pandas()
    result_df.to_pickle(model_args.evaluation_path)
    logging.info("Evaluation is done.")


def length_limited_example(example, tokenizer, max_length, sep="\n"):
    """
    컨텍스트 윈도우 크기에 맞게 텍스트를 준비
    1. next, prev 순으로 연결하고, sep로 분리하고, 토크나이징
    2. 순서대로 길이가 max_length 넘기 직전까지 조사
    3. 조사된 시점까지를 취합하여, 역순으로 정리하면

    (과거 부터 최신까지 순차적으로 기록된 prev) next 까지 연결된 텍스트가 나옴
    """

    # 0->"## 다음 구매 상품 예측", 2->"## 주문히스토리"
    # anchor_indices = [0, 2]

    in_strings = "\n".join((example["next"], example["prev"])).split(sep)

    input_ids = tokenizer.batch_encode_plus(in_strings, add_special_tokens=False)[
        "input_ids"
    ]
    length_s = [len(x) for x in input_ids]

    cum_length = 0
    for i, l in enumerate(length_s):
        cum_length += l
        if cum_length > max_length:
            break

    sub_strings = in_strings[:i]
    sub_strings[0], sub_strings[1] = sub_strings[1], sub_strings[0]  # swap
    sub_strings.append(sub_strings.pop(2))  # pop and append it to the end

    return "\n".join(sub_strings[: i + 1][::-1])


def build_dataset_from_rawdataset(
    model_args: ScriptArguments, training_args: RewardConfig
):
    logging.info("Building dataset...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )

    df = pd.read_csv(
        os.path.join(model_args.raw_data_path, f"{model_args.build_split}.csv")
    )

    pos_neg_group_size = model_args.num_negatives + 1

    assert (
        len(df) % pos_neg_group_size == 0
    ), f"The number of examples should be multiple of {pos_neg_group_size}."

    examples = []
    ids = []
    labels = []

    for i in tqdm(
        range(0, len(df), pos_neg_group_size), total=len(df) // pos_neg_group_size
    ):
        pos_example, neg_examples = df.iloc[i], df.iloc[i + 1 : i + pos_neg_group_size]
        all_labels_zero = (neg_examples["label"] == 0).all()
        assert (
            pos_example["label"] == 1 and all_labels_zero
        ), "The pos_example should has label 1 and All neg_examples have label 0."

        pos_text = length_limited_example(
            pos_example, tokenizer, training_args.max_length
        )
        examples.append(pos_text)
        labels.append(1)
        ids.append(i // pos_neg_group_size)
        for _, neg_example in neg_examples.iterrows():
            neg_text = length_limited_example(
                neg_example, tokenizer, training_args.max_length
            )
            examples.append(neg_text)
            labels.append(0)
            ids.append(i // pos_neg_group_size)

    Dataset.from_dict({"text": examples, "id": ids, "label": labels}).save_to_disk(
        os.path.join(model_args.dataset_path, f"{model_args.build_split}")
    )

    # dataset = load_dataset("katchers", data_files=model_args.raw_data_path)
    # dataset = dataset["train"]
    # dataset = dataset.map(
    #     lambda x: {"text": x["text"]},
    #     remove_columns=dataset.column_names,
    #     num_proc=model_args.num_workers,
    # )
    # dataset.save_to_disk("/Jupyter/dev_src/rec-sys-by-llm/data/katchers/concurrent")
    logging.info("Dataset built.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    evaluate()

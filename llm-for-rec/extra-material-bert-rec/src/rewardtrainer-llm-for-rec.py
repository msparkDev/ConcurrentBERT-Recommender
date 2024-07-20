import os
from dataclasses import dataclass, field
from typing import Optional
import re
import pandas as pd

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
    logging as hf_logging,
)
import logging
from trl import RewardConfig, RewardTrainer

# from trl.import_utils import is_xpu_available
from trl.trainer import ConstantLengthDataset


@dataclass
class ScriptArguments:
    cache_dir: Optional[str] = field(
        default="/Jupyter/huggingface/.cache", metadata={"help": "the cache dir"}
    )

    model_name: Optional[str] = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={"help": "the model name"},
    )

    raw_data_path: Optional[str] = field(
        default="/Jupyter/dev_src//Jupyter/dev_src/rec-sys-by-llm/data/katchers/concurrent",
        metadata={"help": "the raw data path"},
    )

    build_dataset: Optional[bool] = field(
        default=False, metadata={"help": "whether to build the dataset"}
    )

    build_split: Optional[str] = field(
        default="train", metadata={"help": "the split to build the dataset"}
    )

    dataset_path: Optional[str] = field(
        default="/Jupyter/dev_src//Jupyter/dev_src/rec-sys-by-llm/datasets/katchers/concurrent",
        metadata={"help": "the raw data path"},
    )

    seq_length: Optional[int] = field(
        default=1024, metadata={"help": "the sequence length"}
    )

    num_workers: Optional[int] = field(
        default=8, metadata={"help": "the number of workers"}
    )

    model_type: Optional[str] = field(
        default="llama",
        metadata={"help": "You should choose one of mistral, llama2, phi2, midm"},
    )

    packing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use packing for SFTTrainer"}
    )

    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "down_proj",
                "up_proj",
                "gate_proj",
            ],
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["scores"],
        )
    )

    merge_with_final_checkpoint: Optional[bool] = field(
        default=False, metadata={"help": "Do only merge with final checkpoint"}
    )


def build_peft_config(peft_config: LoraConfig, model_type):
    """peft_config에 따라 target_modules를 설정한다."""

    if model_type == "mistral" or model_type == "llama2" or model_type == "gemma":
        peft_config.target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "down_proj",
            "up_proj",
            "gate_proj",
        ]
    elif model_type == "phi2":
        peft_config.target_modules = (
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "dense",
                "fc1",
                "fc2",
            ],
        )
    elif model_type == "midm":
        peft_config.target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return peft_config


def chars_token_ratio(dataset, tokenizer, prepare_sample_text, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def preprocess_function(examples, tokenizer, max_length):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }

    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(
            chosen + tokenizer.eos_token, truncation=True, max_length=max_length
        )
        tokenized_rejected = tokenizer(
            rejected + tokenizer.eos_token, truncation=True, max_length=max_length
        )

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(
            tokenized_rejected["attention_mask"]
        )

    return new_examples


############
def train():
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
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if model_args.build_dataset:
        # 데이터셋 빌드만 하고 종료
        build_dataset_from_rawdataset(model_args, training_args)
        return

    if model_args.merge_with_final_checkpoint:
        # 머지만 하고 종료
        # output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
        output_dir = training_args.output_dir
        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            # repo_type="model",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True,
            # use_auth_token=True,
        )
        model = model.merge_and_unload()

        for param in model.parameters():
            param.data = param.data.contiguous()

        output_merged_dir = os.path.join(
            training_args.output_dir, "final_merged_checkpoint"
        )
        model.save_pretrained(output_merged_dir, safe_serialization=True)
        logging.info("Merge and Safe Seirialization are Done.")
        return

    if training_args.group_by_length and model_args.packing:
        raise ValueError("Cannot use both packing and group by length")

    # 최신 버전에서 해결됐는지 확인 필요
    # `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
    # `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
    # if training_args.gradient_checkpointing:
    #     raise ValueError("gradient_checkpointing not supported")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name,
        quantization_config=bnb_config,
        device_map="auto",  # {"": Accelerator().local_process_index},
        num_labels=1,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )
    base_model.config.use_cache = False

    peft_config = model_args.peft_config
    peft_config = build_peft_config(peft_config, model_args.model_type)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # training_args = model_args.training_args

    tokenize_function = lambda examples: preprocess_function(
        examples, tokenizer=tokenizer, max_length=training_args.max_length
    )

    train_dataset = load_from_disk(os.path.join(model_args.dataset_path, "train"))

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )

    train_dataset = train_dataset.shuffle()

    eval_dataset = load_from_disk(os.path.join(model_args.dataset_path, "val"))

    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )

    trainer = RewardTrainer(
        model=base_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    logging.info(f"Start Fine Tuning... with {model_args.model_type}")
    trainer.train()
    trainer.save_model(training_args.output_dir)

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    logging.info(f"final checkpoint saved to {output_dir}")
    trainer.model.save_pretrained(output_dir)

    logging.info("Fine Tuning is done.")


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
        model_args.model_name,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )

    df = pd.read_csv(
        os.path.join(model_args.raw_data_path, f"{model_args.build_split}.csv")
    )

    assert len(df) % 2 == 0, "The number of examples should be even."

    pos_examples, neg_examples = [], []

    for i in tqdm(range(0, len(df), 2), total=len(df) // 2):
        pos_example, neg_example = df.iloc[i], df.iloc[i + 1]
        assert (
            pos_example["label"] == 1 and neg_example["label"] == 0
        ), "The label should be 1 and 0."
        pos_text = length_limited_example(
            pos_example, tokenizer, training_args.max_length
        )
        neg_text = length_limited_example(
            neg_example, tokenizer, training_args.max_length
        )
        pos_examples.append(pos_text)
        neg_examples.append(neg_text)

    Dataset.from_dict({"chosen": pos_examples, "rejected": neg_examples}).save_to_disk(
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
    train()

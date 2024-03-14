import torch
from datasets import Dataset, load_dataset
from transformers import BertTokenizer, BertForNextSentencePrediction, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# Tokenizes the batch for BERT input
def tokenize_function(examples):
    return tokenizer(examples["prev"], examples["next"], padding=True, truncation=True)

# Computes metrics for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# Loads and tokenizes the dataset from specified file paths
def load_and_tokenize_data(tokenizer, file_paths):
    data_files = {"train": file_paths["train"], "validation": file_paths["val"]}
    dataset = load_dataset("csv", data_files=data_files)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

# Initializes the Trainer object with the given model, datasets, tokenizer, and training arguments
def initialize_trainer(model, train_dataset, eval_dataset, tokenizer, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    return trainer

file_paths = {
    "train": 'data/BERT_ConcurrentPurchases/trainForBERT_WCP.csv',
    "val": 'data/BERT_ConcurrentPurchases/valForBERT_WCP.csv'
}

model_ckpt = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_ckpt)

# Load and tokenize the dataset
tokenized_datasets = load_and_tokenize_data(tokenizer, file_paths)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForNextSentencePrediction.from_pretrained(model_ckpt, num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir="YourUsername/ConcurrentPurchasesBERT-UCIRetailTuned",  # Hugging Face Hub 사용자 이름으로 변경
    num_train_epochs=3,
    learning_rate=2e-5,
    auto_find_batch_size=True,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
<<<<<<< HEAD
    save_strategy="epoch",
=======
    save_strategy="epoch",  
>>>>>>> 483a1cadc622a1bc629429d711832dd12e1a0f54
    push_to_hub=True,
)

# Initialize and run the trainer
trainer = initialize_trainer(model, tokenized_datasets["train"], tokenized_datasets["validation"], tokenizer, training_args)
<<<<<<< HEAD
trainer.train()  # Starts the training process
=======
trainer.train()  # Starts the training process
>>>>>>> 483a1cadc622a1bc629429d711832dd12e1a0f54

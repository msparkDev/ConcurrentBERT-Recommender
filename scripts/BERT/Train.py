import torch
from datasets import Dataset, load_dataset
from transformers import BertTokenizer, BertForNextSentencePrediction, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# Tokenizes batches of text for BERT model input.
def tokenize_function(examples):
    # Applies tokenizer to each pair of sentences in the batch.
    return tokenizer(examples["prev"], examples["next"], padding=True, truncation=True)

# Calculates evaluation metrics for model performance.
def compute_metrics(pred):
    # Extracts the labels and predictions from the output.
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Calculates F1 score and accuracy.
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# Loads and tokenizes dataset from CSV files.
def load_and_tokenize_data(tokenizer, file_paths):
    # Specifies the file paths for training and validation datasets.
    data_files = {"train": file_paths["train"], "validation": file_paths["val"]}
    # Loads the dataset from CSV files.
    dataset = load_dataset("csv", data_files=data_files)
    # Applies tokenization function to the dataset.
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

# Initializes the trainer with the model and datasets.
def initialize_trainer(model, train_dataset, eval_dataset, tokenizer, training_args):
    # Creates a Trainer instance with the specified parameters.
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    return trainer

# File paths to the training and validation datasets.
file_paths = {
    "train": 'data/BERT/Concurrent/train.csv',
    "val": 'data/BERT/Concurrent/val.csv'
}

# Loads the tokenizer for the specified BERT model checkpoint.
model_ckpt = 'google-bert/bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_ckpt)

# Loads and tokenizes the dataset using the specified file paths.
tokenized_datasets = load_and_tokenize_data(tokenizer, file_paths)

# Prepares the model and device for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForNextSentencePrediction.from_pretrained(model_ckpt, num_labels=2).to(device)

# Configures training arguments for the Trainer.
training_args = TrainingArguments(
    output_dir="YourUsernameHere/ConcPurcBERT-UCIRetail",  # Customize with your Hugging Face username.
    num_train_epochs=10,
    learning_rate=2e-5,
    auto_find_batch_size=True,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    push_to_hub=True,
    save_strategy="epoch",
    load_best_model_at_end=True,
    log_level="error")

# Initializes the trainer with the model, datasets, tokenizer, and training arguments.
trainer = initialize_trainer(model, tokenized_datasets["train"], tokenized_datasets["validation"], tokenizer, training_args)

# Starts the training process.
trainer.train()

# Pushes the trained model to the Hugging Face Hub.
trainer.push_to_hub(commit_message="Training completed!")

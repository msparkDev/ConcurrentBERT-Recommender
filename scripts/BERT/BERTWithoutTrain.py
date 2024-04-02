import torch
from datasets import Dataset, load_dataset
from transformers import BertTokenizer, BertForNextSentencePrediction, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from BERTWithTrain import *

# File paths to the training and validation datasets.
file_paths = {
    "train": 'data/BERT_SinglePurchases/trainForBERT_WCP.csv',
    "val": 'data/BERT_SinglePurchases/valForBERT_WCP.csv'
}

<<<<<<< HEAD:scripts/BERT/BERTWithoutTrain.py
# Loads the tokenizer for the specified BERT model checkpoint.
=======
# Defines the BERT model checkpoint for tokenizer initialization.
>>>>>>> 7f1ff6c0d40b0dc1a08037ba622bc0ac81e4415a:scripts/BERT_SinglePurchases_Training.py
model_ckpt = 'google-bert/bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_ckpt)

# Loads and tokenizes the dataset using the specified file paths.
tokenized_datasets = load_and_tokenize_data(tokenizer, file_paths)

# Prepares the model and device for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForNextSentencePrediction.from_pretrained(model_ckpt, num_labels=2).to(device)

# Configures training arguments for the Trainer.
training_args = TrainingArguments(
    output_dir="YourUsernameHere/SinglePurchasesBERT-UCIRetailTuned",  # Customize with your Hugging Face username.
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

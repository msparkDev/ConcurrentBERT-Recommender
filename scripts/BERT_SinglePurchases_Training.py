import torch
from datasets import Dataset, load_dataset
from transformers import BertTokenizer, BertForNextSentencePrediction, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from BERT_ConcurrentPurchases_Training import *

# Specifies file paths for training and validation datasets.
file_paths = {
    "train": 'data/BERT_SinglePurchases/trainForBERT_WOCP.csv',
    "val": 'data/BERT_SinglePurchases/valForBERT_WOCP.csv'
}

# Defines the BERT model checkpoint for tokenizer initialization.
model_ckpt = 'google-bert/bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_ckpt)

# Loads and tokenizes the dataset using previously defined functions.
tokenized_datasets = load_and_tokenize_data(tokenizer, file_paths)

# Sets up the device for training, using GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForNextSentencePrediction.from_pretrained(model_ckpt, num_labels=2).to(device)

# Configures training arguments for fine-tuning the model.
training_args = TrainingArguments(
    output_dir="YourUsernameHere/SinglePurchasesBERT-UCIRetailTuned",  # Replace "YourUsernameHere" with your username.
    num_train_epochs=3,
    learning_rate=2e-5,
    auto_find_batch_size=True,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    push_to_hub=True,
    save_strategy="epoch",
    load_best_model_at_end=True,
    log_level="error")

# Initializes the training process with the model, datasets, and specified training arguments.
trainer = initialize_trainer(model, tokenized_datasets["train"], tokenized_datasets["validation"], tokenizer, training_args)

# Begins model training using the Trainer instance.
trainer.train()

# Uploads the trained model to the Hugging Face Hub for sharing and reuse.
trainer.push_to_hub(commit_message="Training completed!")

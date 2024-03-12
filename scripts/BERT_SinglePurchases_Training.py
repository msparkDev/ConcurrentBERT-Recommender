import gc
import torch
from datasets import Dataset, load_dataset
from transformers import BertTokenizer, BertForNextSentencePrediction, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import notebook_login
from BERT_ConcurrentPurchases_Training import tokenize_function, compute_metrics, load_and_tokenize_data, initialize_trainer, 

# Main execution flow
notebook_login()  # Authenticates the user for Hugging Face Hub

file_paths = {
    "train": 'data/BERT_SinglePurchases/trainForBERT_WCP.csv',
    "val": 'data/BERT_SinglePurchases/valForBERT_WCP.csv'
}

model_ckpt = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_ckpt)

# Load and tokenize the dataset
tokenized_datasets = load_and_tokenize_data(tokenizer, file_paths)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForNextSentencePrediction.from_pretrained(model_ckpt, num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir="YourUsernameHere/SinglePurchasesBERT-UCIRetailTuned",  # Change "YourUsernameHere" to your Hugging Face username.
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    push_to_hub=True,
)

# Initialize and run the trainer
trainer = initialize_trainer(model, tokenized_datasets["train"], tokenized_datasets["validation"], tokenizer, training_args)
trainer.train()  # Starts the training process

# Push the trained model to the Hugging Face Hub
trainer.push_to_hub(commit_message="Training completed!")  # Commits the model to the Hugging Face Hub

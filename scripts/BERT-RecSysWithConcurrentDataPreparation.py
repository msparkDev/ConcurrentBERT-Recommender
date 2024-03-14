import torch
from datasets import Dataset, load_dataset
from transformers import BertTokenizer, BertForNextSentencePrediction, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from BERT_ConcurrentPurchases_Training import *

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
    output_dir="YourUsername/SinglePurchasesBERT-UCIRetailTuned",  # Hugging Face Hub 사용자 이름으로 변경
    num_train_epochs=3,
    learning_rate=2e-5,
    auto_find_batch_size=True,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    save_strategy="epoch",
    push_to_hub=True,
)

# Initialize and run the trainer
trainer = initialize_trainer(model, tokenized_datasets["train"], tokenized_datasets["validation"], tokenizer, training_args)
trainer.train()  # Starts the training process
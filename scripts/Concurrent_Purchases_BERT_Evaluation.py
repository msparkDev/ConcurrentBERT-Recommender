import pandas as pd
import numpy as np
import math
from transformers import BertTokenizer, BertForNextSentencePrediction, Trainer
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset

# Function to tokenize and predict the test set using the BERT model
def tokenize_and_predict(test_set):
    """Tokenizes the test set and predicts labels."""
    test_encoded = Dataset.from_pandas(test_set)
    test_encoded = test_encoded.map(lambda batch: tokenizer(batch["prev"], batch["next"], padding=True, truncation=True), batched=True)
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_encoded)
    preds = np.argmax(predictions.predictions, axis=-1)
    return preds, test_encoded['label']

# Function to calculate the Normalized Discounted Cumulative Gain (NDCG)
def calculate_ndcg(group_data, k=10):
    """Calculates NDCG for the given data."""
    group_data = group_data.sort_values(by='score', ascending=False)
    rel = group_data['label'].tolist()
    dcg = sum([rel[i] / math.log2(i+2) for i in range(k)])
    idcg = sum([1 / math.log2(i+2) for i in range(1, min(k, len(rel)) + 1)])
    return dcg / idcg if idcg > 0 else 0

# Initializing the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("MSParkDev/ConcurrentPurchasesBERT-UCIRetailTuned")
model = BertForNextSentencePrediction.from_pretrained("MSParkDev/ConcurrentPurchasesBERT-UCIRetailTuned")

# Loading the test dataset
test_set = pd.read_csv('data/BERT_ConcurrentPurchases/testForBERT_WCP.csv')

# Main execution flow starts here
# Tokenize and predict the test dataset
preds, true_labels = tokenize_and_predict(test_set)

# Calculate performance metrics
accuracy = accuracy_score(true_labels, preds)
f1 = f1_score(true_labels, preds, average="weighted")

# Calculate NDCG values
ndcg5_values = []
ndcg10_values = []

for key, group in test_set.groupby('prev'):
    group_data = group.copy().reset_index(drop=True)
    scores = []
    
    # Score each item in the group using the model
    for i in range(len(group_data)):
        inputs = tokenizer(group_data.loc[i, 'prev'], group_data.loc[i, 'next'], return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        scores.append(outputs.logits[0][1].item() - outputs.logits[0][0].item())
    
    group_data['score'] = scores
    
    # Append NDCG values
    ndcg5_values.append(calculate_ndcg(group_data, 5))
    ndcg10_values.append(calculate_ndcg(group_data, 10))

# Print Metrics
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print("Average NDCG@5: ", np.mean(ndcg5_values))
print("Average NDCG@10: ", np.mean(ndcg10_values))

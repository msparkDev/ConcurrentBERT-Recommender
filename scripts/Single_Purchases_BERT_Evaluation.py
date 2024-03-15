import pandas as pd
import numpy as np
import math
from transformers import BertTokenizer, BertForNextSentencePrediction, Trainer
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from Concurrent_Purchases_BERT_Evaluation import *

# Initialize the tokenizer and model with pretrained weights.
tokenizer = BertTokenizer.from_pretrained("MSParkDev/SinglePurchasesBERT-UCIRetailTuned")
model = BertForNextSentencePrediction.from_pretrained("MSParkDev/SinglePurchasesBERT-UCIRetailTuned")

# Load the test dataset.
test_set = pd.read_csv('data/BERT_SinglePurchases/testForBERT_WOCP.csv')

# Main execution flow for making predictions and calculating metrics.
preds, true_labels = tokenize_and_predict(test_set)

# Calculate and print performance metrics.
accuracy = accuracy_score(true_labels, preds)
f1 = f1_score(true_labels, preds, average="weighted")

ndcg5_values = []
ndcg10_values = []

# Calculate NDCG scores for the test dataset.
for key, group in test_set.groupby('prev'):
    group_data = group.copy().reset_index(drop=True)
    scores = []

    # Calculate scores for each row in the dataset to evaluate the model's prediction.
    for i in range(len(group_data)):
        inputs = tokenizer(group_data.loc[i, 'prev'], group_data.loc[i, 'next'], return_tensors="pt", padding=True,
                           truncation=True)
        outputs = model(**inputs)
        scores.append(outputs.logits[0][1].item() - outputs.logits[0][0].item())

    group_data['score'] = scores
    ndcg5_values.append(calculate_ndcg(group_data, 5))
    ndcg10_values.append(calculate_ndcg(group_data, 10))

# Output the calculated accuracy, F1 score, and average NDCG scores.
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print("Average NDCG@5: ", np.mean(ndcg5_values))
print("Average NDCG@10: ", np.mean(ndcg10_values))

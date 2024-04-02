import pandas as pd
import numpy as np
import math
from transformers import BertTokenizer, BertForNextSentencePrediction, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from BERTWithEval import *

# Initialize the tokenizer and model
model_checkpoint = "YourUserNameHere/SingPurcBERT-UCIRetail"
tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
model = BertForNextSentencePrediction.from_pretrained(model_checkpoint)

# Load the test dataset
test_set = pd.read_csv('data/BERT_SinglePurchases/testForBERT_WOCP.csv')
test_set = test_set.drop_duplicates(subset=['prev', 'label'])

# Main execution flow
preds, true_labels = tokenize_and_predict(test_set)

# Compute performance metrics
accuracy = accuracy_score(true_labels, preds)
f1 = f1_score(true_labels, preds, average="weighted")
print(f"Accuracy: {accuracy}\nF1 Score: {f1}")

# Calculate NDCG values
ndcg5_values, ndcg10_values = [], []
for key, group in test_set.groupby('prev'):
    group_data = group.reset_index(drop=True)
    scores = [
        model(**tokenizer(group_data.loc[i, 'prev'], group_data.loc[i, 'next'], return_tensors="pt", padding=True,
                          truncation=True)).logits[0][1].item() -
        model(**tokenizer(group_data.loc[i, 'prev'], group_data.loc[i, 'next'], return_tensors="pt", padding=True,
                          truncation=True)).logits[0][0].item()
        for i in range(len(group_data))
    ]
    group_data['score'] = scores
    ndcg5_values.append(calculate_ndcg(group_data, 5))
    ndcg10_values.append(calculate_ndcg(group_data, 10))

# Output average NDCG scores
print(f"Average NDCG@5: {np.mean(ndcg5_values)}\nAverage NDCG@10: {np.mean(ndcg10_values)}")

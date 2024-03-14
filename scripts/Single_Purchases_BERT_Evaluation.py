import pandas as pd
import numpy as np
import math
from transformers import BertTokenizer, BertForNextSentencePrediction, Trainer
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from Concurrent_Purchases_BERT_Evaluation import *

# Initializing the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("MSParkDev/SinglePurchasesBERT-UCIRetailTuned")
model = BertForNextSentencePrediction.from_pretrained("MSParkDev/SinglePurchasesBERT-UCIRetailTuned")

# Loading the test dataset
test_set = pd.read_csv('data/BERT_SinglePurchases/testForBERT_WOCP.csv')

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

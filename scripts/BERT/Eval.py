import pandas as pd
import numpy as np
import math
from transformers import BertTokenizer, BertForNextSentencePrediction, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

# Initialize the tokenizer and model
model_checkpoint = "YourUsernameHere/ConcPurcBERT-UCIRetail"
tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
model = BertForNextSentencePrediction.from_pretrained(model_checkpoint)


def tokenize_and_predict(test_set):
    """
    Tokenizes the test set and predicts the next sentence classification.

    Parameters:
    - test_set: A DataFrame with columns 'prev' and 'next' for the text to be classified, and 'label'.

    Returns:
    - preds: Predicted classes (0 or 1) for each sample.
    - true_labels: Actual classes for each sample.
    """
    test_encoded = Dataset.from_pandas(test_set)
    test_encoded = test_encoded.map(
        lambda batch: tokenizer(batch["prev"], batch["next"], padding=True, truncation=True),
        batched=True
    )
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_encoded)
    preds = np.argmax(predictions.predictions, axis=-1)
    return preds, test_encoded['label']


def calculate_ndcg(group_data, k=10):
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG) for a group of predictions.

    Parameters:
    - group_data: A DataFrame with columns 'score' and 'label', sorted by 'score'.
    - k: The number of items to consider in the calculation.

    Returns:
    - The NDCG score for the group.
    """
    group_data = group_data.sort_values(by='score', ascending=False)
    rel = group_data['label'].tolist()
    dcg = sum([rel[i] / math.log2(i + 2) for i in range(k)])
    idcg = sum([sorted(rel, reverse=True)[i] / math.log2(i + 2) for i in range(min(k, len(rel)))])
    return dcg / idcg if idcg > 0 else 0


# Load the test dataset
test_set = pd.read_csv('data/BERT/Concurrent/test.csv')
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

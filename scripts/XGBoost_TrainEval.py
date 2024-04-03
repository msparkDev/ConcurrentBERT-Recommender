import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

from scripts.BERT.Eval import calculate_ndcg
from scripts.DeepFM.TrainEval import load_data, split_data

# Data Loading and Preprocessing
def load_and_preprocess_data():
    # Fetch and preprocess data
    data = fetch_ucirepo(id=352).data.original
    return data

def encode_features(data, feature_cols):
    # Encode categorical features
    label_encoder = LabelEncoder()
    for col in feature_cols:
        data[col] = label_encoder.fit_transform(data[col])
    return data


# Model Training and Evaluation
def train_xgb_model(X_train, y_train, X_val, y_val):
    # Train XGBoost model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    return model

def evaluate_model(model, X_test, y_test):
    # Evaluate the model on test dataset
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    print(f"F1 Score: {f1}\nAccuracy: {accuracy}")
    return predictions

def calculate_ndcg_scores(test_data, predictions, k_values=[5, 10]):
    # Calculate and print NDCG scores for multiple k values
    test_data['score'] = predictions
    test_data['group'] = np.arange(len(test_data)) // 50  # Example group assignment
    grouped_test = test_data.groupby('group')
    for k in k_values:
        ndcg_scores = grouped_test.apply(lambda x: calculate_ndcg(x, k=k))
        print(f"Mean NDCG@{k}: {ndcg_scores.mean()}")

# Main Execution Flow
combined_data = load_data()
sparse_features = ['C' + str(i) for i in range(1, 11)]  # Define sparse features
    
combined_data = encode_features(combined_data, sparse_features)  # Encode sparse features
train_data, val_data, test_data = split_data(combined_data)  # Split the combined dataset
    
X_train, y_train = train_data.drop(columns=['label']), train_data['label']
X_val, y_val = val_data.drop(columns=['label']), val_data['label']
X_test, y_test = test_data.drop(columns=['label']), test_data['label']
    
model = train_xgb_model(X_train, y_train, X_val, y_val)  # Train model
predictions = evaluate_model(model, X_test, y_test)  # Evaluate model
    
calculate_ndcg_scores(test_data, predictions)  # Calculate NDCG scores

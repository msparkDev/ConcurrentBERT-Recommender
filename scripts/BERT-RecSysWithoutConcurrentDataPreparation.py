import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from transformers import BertTokenizer
from BERT-RecSysWithConcurrentDataPreparation import (
    split_user_data, format_next_purchase, get_longer_text,
    add_text_if_fits, generate_dataset, process_dataset)

# Define functions for data preprocessing

def compile_order_history(user_prompt, tokenizer, item_max):
    """
    Compile order history into a text string.
    
    Parameters:
    - user_prompt: DataFrame containing user's purchase history.
    - tokenizer: Tokenizer object for encoding the text.
    - item_max: Maximum number of items to consider.
    
    Returns:
    - A string representing the compiled order history.
    """
    user_text = "## order history"  # Initialize the order history text with a heading
    
    user_prompt['InvoiceDate'] = pd.to_datetime(
        user_prompt['InvoiceDate'], format='%m/%d/%Y %H:%M')  # Convert 'InvoiceDate' to datetime
    
    user_prompt = user_prompt.sort_values(
        'InvoiceDate', ascending=False)  # Sort data by 'InvoiceDate' to prioritize recent orders
    
    for _, row in user_prompt.iterrows():  # Compile individual purchase details
        detail = f"\nOrder on {row['InvoiceDate'].strftime('%m/%d/%Y %H:%M')}: {row['Description']},"
        new_user_text = add_text_if_fits(user_text, detail, tokenizer, item_max)
        if new_user_text is not None:
            user_text = new_user_text
        else:
            return user_text
    return user_text

# Main data processing flow

data_dir = "data/BERT_ConcurrentPurchases"  # Directory for the concurrent purchase datasets

# Construct file paths for the datasets
train_data_path = os.path.join(data_dir, 'train_data.csv')
val_data_path = os.path.join(data_dir, 'validation_data.csv')
test_data_path = os.path.join(data_dir, 'test_data.csv')

# Read datasets into pandas DataFrames
train_data = pd.read_csv(train_data_path)
val_data = pd.read_csv(val_data_path)
test_data = pd.read_csv(test_data_path)

# Construct file paths for the negative sample datasets
negative_train_path = os.path.join(data_dir, 'negative_train.csv')
negative_val_path = os.path.join(data_dir, 'negative_val.csv')
negative_test_path = os.path.join(data_dir, 'negative_test.csv')

# Read negative sample datasets into pandas DataFrames
negative_train = pd.read_csv(negative_train_path)
negative_val = pd.read_csv(negative_val_path)
negative_test = pd.read_csv(negative_test_path)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Process and save the datasets for training, validation, and testing
train_df = process_dataset(train_data, train_data.groupby('CustomerID'), negative_train, "train")
val_df = process_dataset(val_data, val_data.groupby('CustomerID'), negative_val, "validation")
test_df = process_dataset(test_data, test_data.groupby('CustomerID'), negative_test, "test")

new_data_dir = "data/BERT_SinglePurchases"  # Directory for single purchase datasets

os.makedirs(new_data_dir, exist_ok=True)  # Ensure the directory exists; create it if it doesn't

# Save the processed datasets to the new directory
train_df.to_csv(os.path.join(new_data_dir, 'trainForBERT_WOCP.csv'), index=False)
val_df.to_csv(os.path.join(new_data_dir, 'valForBERT_WOCP.csv'), index=False)
test_df.to_csv(os.path.join(new_data_dir, 'testForBERT_WOCP.csv'), index=False)

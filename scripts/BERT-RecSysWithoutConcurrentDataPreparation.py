import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from transformers import BertTokenizer
from BERT-RecSysWithConcurrentDataPreparation import split_user_data, format_next_purchase, get_longer_text, add_text_if_fits, generate_dataset, process_dataset

# Define functions for data preprocessing and negative sampling.

def compile_order_history(user_prompt, tokenizer, item_max):
    # Initialize the order history text with a heading.
    user_text = "## order history"
    
    # Convert 'InvoiceDate' to datetime format for correct sorting
    user_prompt['InvoiceDate'] = pd.to_datetime(user_prompt['InvoiceDate'], format='%m/%d/%Y %H:%M')
    
    # Sort the data by 'InvoiceDate' in descending order to prioritize recent orders.
    user_prompt = user_prompt.sort_values('InvoiceDate', ascending=False)
    
    # Iterate over each row in the sorted data to compile individual purchase details.
    for _, row in user_prompt.iterrows():
        # Format the order detail with the date and description of the item.
        temp = f"\nOrder on {row['InvoiceDate'].strftime('%m/%d/%Y %H:%M')}: {row['Description']},"
        
        # Check if adding this order detail exceeds the token limit for the compiled text.
        new_user_text = add_text_if_fits(user_text, temp, tokenizer, item_max)
        if new_user_text is not None:
            user_text = new_user_text  # Update user_text with the new order detail
        else:
            return user_text  # Return the text if adding another detail would exceed the limit
                
    return user_text  # Return the complete order history text

# Main Data Processing Flow

data_dir = "data/BERT_ConcurrentPurchases"

# Use os.path.join to construct the file path
train_data_path = os.path.join(data_dir, 'train_data.csv')
val_data_path = os.path.join(data_dir, 'validation_data.csv')
test_data_path = os.path.join(data_dir, 'test_data.csv')

# Reading the CSV files into pandas DataFrames
train_data = pd.read_csv(train_data_path)
val_data = pd.read_csv(val_data_path)
test_data = pd.read_csv(test_data_path)

# Use os.path.join to construct the file path
negative_train_path = os.path.join(data_dir, 'negative_train.csv')
negative_val_path = os.path.join(data_dir, 'negative_val.csv')
negative_test_path = os.path.join(data_dir, 'negative_test.csv')

# Reading the CSV files into pandas DataFrames
negative_train = pd.read_csv(negative_train_path)
negative_val = pd.read_csv(negative_val_path)
negative_test = pd.read_csv(negative_test_path)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Process and save the datasets for training, validation, and testing
train_df = process_dataset(train_data, train_data.groupby('CustomerID'), negative_train, "train")
val_df = process_dataset(validation_data, validation_data.groupby('CustomerID'), negative_val, "validation")
test_df = process_dataset(test_data, test_data.groupby('CustomerID'), negative_test, "test")

new_data_dir = "data/BERT_SinglePurchases"

train_df.to_csv(os.path.join(new_data_dir, 'trainForBERT_WOCP.csv'), index=False)
val_df.to_csv(os.path.join(new_data_dir, 'valForBERT_WOCP.csv'), index=False)
test_df.to_csv(os.path.join(new_data_dir, 'testForBERT_WOCP.csv'), index=False)

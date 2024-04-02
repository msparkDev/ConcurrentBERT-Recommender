import os
import pandas as pd
from transformers import BertTokenizer
from BERTWithDataPrep import *


def compile_order_history(user_prompt, tokenizer, item_max):
    """
    Compiles a user's order history into a descriptive text string.

    Parameters:
    - user_prompt: DataFrame containing the user's purchase history.
    - tokenizer: BertTokenizer instance for text processing.
    - item_max: Maximum token count for BERT input.

    Returns:
    - A string that represents the user's order history, formatted for BERT input.
    """
    # Initialize order history with a header
    user_text = "## order history"

    # Ensure 'InvoiceDate' is in datetime format for sorting
    user_prompt['InvoiceDate'] = pd.to_datetime(user_prompt['InvoiceDate'], format='%m/%d/%Y %H:%M')
    user_prompt = user_prompt.sort_values('InvoiceDate', ascending=False)

    # Compile order details, checking token limit at each addition
    for _, row in user_prompt.iterrows():
        detail = f"\nOrder on {row['InvoiceDate'].strftime('%m/%d/%Y %H:%M')}: {row['Description']},"
        potential_text = add_text_if_fits(user_text, detail, tokenizer, item_max)
        user_text = potential_text if potential_text is not None else user_text

    return user_text


# Main Data Processing Script

# Define directories for data storage and retrieval
data_dir = "data/BERT_ConcurrentPurchases"
new_data_dir = "data/BERT_SinglePurchases"

# Ensure the new data directory exists
os.makedirs(new_data_dir, exist_ok=True)

# Load datasets and negative samples into DataFrames
train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
val_data = pd.read_csv(os.path.join(data_dir, 'val_data.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
negative_train = pd.read_csv(os.path.join(data_dir, 'negative_train.csv'))
negative_val = pd.read_csv(os.path.join(data_dir, 'negative_val.csv'))
negative_test = pd.read_csv(os.path.join(data_dir, 'negative_test.csv'))

# Initialize BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Process the datasets for training, validation, and testing
train_df = process_dataset(train_data, train_data.groupby('CustomerID'), negative_train, "train")
val_df = process_dataset(val_data, val_data.groupby('CustomerID'), negative_val, "validation")
test_df = process_dataset(test_data, test_data.groupby('CustomerID'), negative_test, "test")

# Save the processed datasets for model training
train_df.to_csv(os.path.join(new_data_dir, 'trainForBERT_WOCP.csv'), index=False)
val_df.to_csv(os.path.join(new_data_dir, 'valForBERT_WOCP.csv'), index=False)
test_df.to_csv(os.path.join(new_data_dir, 'testForBERT_WOCP.csv'), index=False)

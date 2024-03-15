import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from transformers import BertTokenizer
from BERT

-RecSysWithConcurrentDataPreparation
import

(
    split_user_data, format_next_purchase, get_longer_text,
    add_text_if_fits, generate_dataset, process_dataset)


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

    # Convert 'InvoiceDate' to datetime for proper sorting and manipulation
    user_prompt['InvoiceDate'] = pd.to_datetime(
        user_prompt['InvoiceDate'], format='%m/%d/%Y %H:%M')

    # Sort the user's purchases by date, with the most recent purchases first
    user_prompt = user_prompt.sort_values(
        'InvoiceDate', ascending=False)

    # Iterate through each purchase in the sorted history
    for _, row in user_prompt.iterrows():
        # Format the text for each purchase
        detail = f"\nOrder on {row['InvoiceDate'].strftime('%m/%d/%Y %H:%M')}: {row['Description']},"
        # Attempt to add this purchase's details to the compiled history
        new_user_text = add_text_if_fits(user_text, detail, tokenizer, item_max)
        # If the addition fits within the token limit, update the history
        if new_user_text is not None:
            user_text = new_user_text
        else:
            # If adding another purchase would exceed the token limit, return the current history
            return user_text
    return user_text


# Main data processing flow

# Specify the directory where the datasets are located
data_dir = "data/BERT_ConcurrentPurchases"

# Construct paths to the dataset files for training, validation, and testing
train_data_path = os.path.join(data_dir, 'train_data.csv')
val_data_path = os.path.join(data_dir, 'validation_data.csv')
test_data_path = os.path.join(data_dir, 'test_data.csv')

# Load the datasets into pandas DataFrames
train_data = pd.read_csv(train_data_path)
val_data = pd.read_csv(val_data_path)
test_data = pd.read_csv(test_data_path)

# Paths to the negative samples datasets for training, validation, and testing
negative_train_path = os.path.join(data_dir, 'negative_train.csv')
negative_val_path = os.path.join(data_dir, 'negative_val.csv')
negative_test_path = os.path.join(data_dir, 'negative_test.csv')

# Load the negative samples into pandas DataFrames
negative_train = pd.read_csv(negative_train_path)
negative_val = pd.read_csv(negative_val_path)
negative_test = pd.read_csv(negative_test_path)

# Initialize the BERT tokenizer for text preprocessing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Process the datasets for training, validation, and testing, incorporating negative samples
train_df = process_dataset(train_data, train_data.groupby('CustomerID'), negative_train, "train")
val_df = process_dataset(val_data, val_data.groupby('CustomerID'), negative_val, "validation")
test_df = process_dataset(test_data, test_data.groupby('CustomerID'), negative_test, "test")

# Define a new directory for saving the processed datasets
new_data_dir = "data/BERT_SinglePurchases"

# Ensure the new directory exists; create it if it doesn't
os.makedirs(new_data_dir, exist_ok=True)

# Save the processed datasets to files in the new directory
train_df.to_csv(os.path.join(new_data_dir, 'trainForBERT_WOCP.csv'), index=False)
val_df.to_csv(os.path.join(new_data_dir, 'valForBERT_WOCP.csv'), index=False)
test_df.to_csv(os.path.join(new_data_dir, 'testForBERT_WOCP.csv'), index=False)

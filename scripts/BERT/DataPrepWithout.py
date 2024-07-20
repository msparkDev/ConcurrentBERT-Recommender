import os
import pandas as pd
from transformers import BertTokenizer
from scripts.BERT.DataPrepWith import *


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
    user_text = "## 주문 히스토리"
    user_prompt = user_prompt.sort_values('initial_paid_at', ascending=False)
    user_prompt_grouped = user_prompt.groupby('initial_paid_at', sort=False)

    # Compile order details, checking token limit at each addition
    order_index = 0
    for _, row in user_prompt.iterrows():
        if order_index > 0:
            detail = ', '
        else:
            detail = ''

        if pd.notna(row.attribute_values):
            detail += f'\n{row["initial_paid_at"]}의 주문 내역: {row["category_name"]} - {row["product_name"]} {row["attribute_values"]}'
        else:
            detail += f'\n{row["initial_paid_at"]}의 주문 내역: {row["category_name"]} - {row["product_name"]}'

        potential_text = add_text_if_fits(user_text, detail, tokenizer, item_max)
        user_text = potential_text if potential_text is not None else user_text

    return user_text


# Main Data Processing Script

# Define directories for data storage and retrieval
data_dir = "data/BERT/Concurrent"
new_data_dir = "data/BERT/Single"

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
train_df = process_dataset(train_data, train_data.groupby('user_id'), negative_train, "train")
val_df = process_dataset(val_data, val_data.groupby('user_id'), negative_val, "validation")
test_df = process_dataset(test_data, test_data.groupby('user_id'), negative_test, "test")

# Save the processed datasets for model training
train_df.to_csv(os.path.join(new_data_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(new_data_dir, 'val.csv'), index=False)
test_df.to_csv(os.path.join(new_data_dir, 'test.csv'), index=False)

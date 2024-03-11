import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from transformers import BertTokenizer
from BERT_RecSysWithConcurrentDataPreparation import split_user_data, format_next_purchase, get_longer_text, add_text_if_fits, generate_dataset

# Initialize tokenizer globally to be used in the compile_order_history function
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def compile_order_history(user_prompt, tokenizer, item_max):
    """
    Compile order history without considering concurrent purchases.
    """
    user_text = "## order history"
    user_prompt = user_prompt.groupby('InvoiceDate')
    
    for date, group in user_prompt:
        # Add order details for each date
        date_text = f'\nOrder details for {date}: '
        user_text = add_text_if_fits(user_text, date_text, tokenizer, item_max)
        if not user_text: return user_text  # Return current text if limit exceeded
        
        for description in group['Description']:
            description_text = description + ', '
            new_user_text = add_text_if_fits(user_text, description_text, tokenizer, item_max)
            if not new_user_text:
                return user_text  # Return the current text if limit exceeded
            else:
                user_text = new_user_text  # Update the text with the new addition
                
    return user_text

# Data directory for concurrent purchases
data_dir_concurrent = "data/BERT_ConcurrentPurchases"

# Load data for concurrent purchases
train_data = pd.read_csv(os.path.join(data_dir_concurrent, 'train_data.csv'))
val_data = pd.read_csv(os.path.join(data_dir_concurrent, 'validation_data.csv'))
test_data = pd.read_csv(os.path.join(data_dir_concurrent, 'test_data.csv'))

# Load negative samples
negative_train = pd.read_csv(os.path.join(data_dir_concurrent, 'negative_train.csv'))
negative_val = pd.read_csv(os.path.join(data_dir_concurrent, 'negative_val.csv'))
negative_test = pd.read_csv(os.path.join(data_dir_concurrent, 'negative_test.csv'))

# Group the data by CustomerID for processing
grouped_train = train_data.groupby('CustomerID')
grouped_val = val_data.groupby('CustomerID')
grouped_test = test_data.groupby('CustomerID')

# Data directory for datasets without considering concurrent purchases
data_dir_no_concurrent = "data/BERT_SinglePurchases"
os.makedirs(data_dir_no_concurrent, exist_ok=True)

# Generate and save datasets
for mode, dataset, grouped_data in [("train", train_data, grouped_train),
                                     ("validation", val_data, grouped_val),
                                     ("test", test_data, grouped_test)]:
    group_keys = grouped_data.groups.keys()
    sentence_prev, sentence_next, labels = generate_dataset(group_keys, grouped_data,
                                                            negative_train if mode == "train" else (
                                                            negative_val if mode == "validation" else negative_test),
                                                            mode)

    df = pd.DataFrame({'prev': sentence_prev, 'next': sentence_next, 'label': labels})
    # Save the datasets without considering concurrent purchases in the correct directory
    df.to_csv(os.path.join(data_dir_no_concurrent, f'{mode}ForBERT_WOCP.csv'), index=False)

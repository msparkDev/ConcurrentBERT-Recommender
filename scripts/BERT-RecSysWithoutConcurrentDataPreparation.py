import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from transformers import BertTokenizer
from BERT-RecSysWithConcurrentDataPreparation import split_user_data, format_next_purchase, get_longer_text, add_text_if_fits, generate_dataset

# Define functions for data preprocessing and negative sampling.

def compile_order_history(user_prompt, tokenizer, item_max):
    """
    Compiles a user's order history into a coherent text, respecting the token limit.
    """
    user_text = "## order history"
    user_prompt['InvoiceDate'] = pd.to_datetime(user_prompt['InvoiceDate'], format='%m/%d/%Y %H:%M')
    user_prompt = user_prompt.sort_values('InvoiceDate', ascending=False)
    user_prompt_grouped = user_prompt.groupby('InvoiceDate', sort=False)

    for date, group in user_prompt_grouped:
        temp = f'\nOrder details for {date.strftime("%m/%d/%Y %H:%M")}'
        if len(group) > 1:
            temp += ' [Concurrent Purchase]'
        temp += ': '
        for description in group['Description']:
            temp += description + ', '
        new_user_text = add_text_if_fits(user_text, temp, tokenizer, item_max)
        if new_user_text is not None:
            user_text = new_user_text
        else:
            return user_text
    return user_text

def generate_dataset(group_keys, grouped_data, negative_data, mode):
    """
    Generates the dataset for model training, validation, or testing by creating sequences from user purchase history.
    """
    sentence_prev = []
    sentence_next = []
    labels = []
    
    for key in group_keys:
        group_data = grouped_data.get_group(key).copy()
        user_prompt, user_positive = split_user_data(group_data)
        user_negative = negative_data[negative_data.CustomerID == key]
        
        item_pos = format_next_purchase(user_positive)
        sentence_next.append(item_pos)
        labels.append(1)
        
        if mode in ["train", "validation"]:
            item_neg = format_next_purchase(user_negative)
            sentence_next.append(item_neg)
            labels.append(0)
            
            item_max = get_longer_text(item_pos, item_neg)
            user_text = compile_order_history(user_prompt, tokenizer, item_max)
            for i in range(2):
                sentence_prev.append(user_text)
        elif mode == "test":
            item_max = item_pos
            for i, (index, neg) in enumerate(user_negative.iterrows()):
                item_neg = format_next_purchase(neg)
                sentence_next.append(item_neg)
                labels.append(0)
                item_max = get_longer_text(item_max, item_neg)
            user_text = compile_order_history(user_prompt, tokenizer, item_max)
            for i in range(len(user_negative)):
                sentence_prev.append(user_text)
    return sentence_prev, sentence_next, labels

def prepare_negative_samples(data, unique_user_ids, N, mode='train'):
    """
    Prepares negative samples for the dataset by generating non-purchased product IDs for each user.
    """
    unique_product_ids = set(data['Description'].unique())
    negative_product_id_df = create_negative_samples_dataframe(data, unique_user_ids, unique_product_ids, N)
    xx = data.groupby('Description', as_index=False).last()
    negative_samples_merged = pd.merge(negative_product_id_df, xx, on='Description', how='left').drop(['CustomerID_y'], axis=1)
    negative_samples_merged.rename(columns={'CustomerID_x': 'CustomerID'}, inplace=True)
    return negative_samples_merged

def process_dataset(data, grouped_data, negative_data, mode):
    """
    Processes the dataset for the given mode (train, validation, test) by generating text sequences and labels.
    """
    group_keys = grouped_data.groups.keys()
    sentence_prev, sentence_next, labels = generate_dataset(group_keys, grouped_data, negative_data, mode)
    df = pd.DataFrame({'prev': sentence_prev, 'next': sentence_next, 'label': labels})
    df = df[df.prev != '## order history']
    return df

# Main Data Processing Flow

# Fetch and preprocess data from UCI ML repository
data = fetch_ucirepo(id=352).data.features

# Remove users with only one purchase
data['total_order_cnt'] = data.groupby('CustomerID')['Description'].transform('count')
data = data[data['total_order_cnt'] > 1].drop(['total_order_cnt'], axis=1).reset_index(drop=True)

# Split dataset into training, validation, and test sets
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_val_idx, test_idx = next(gss.split(data, groups=data['CustomerID']))
train_val_data = data.iloc[train_val_idx]
test_data = data.iloc[test_idx]

gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
train_idx, val_idx = next(gss.split(train_val_data, groups=train_val_data['CustomerID']))
train_data = train_val_data.iloc[train_idx]
validation_data = train_val_data.iloc[val_idx]

# Ensure the data directory exists
data_dir = "data/BERT_ConcurrentPurchases"
os.makedirs(data_dir, exist_ok=True)

# Save the split data
train_data.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
validation_data.to_csv(os.path.join(data_dir, 'validation_data.csv'), index=False)
test_data.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)

# Prepare negative samples for each dataset part
negative_train = prepare_negative_samples(train_data, train_data['CustomerID'].unique(), N=1)
negative_val = prepare_negative_samples(validation_data, validation_data['CustomerID'].unique(), N=1)
negative_test = prepare_negative_samples(test_data, test_data['CustomerID'].unique(), N=49)

# Save negative samples to files
negative_train.to_csv(os.path.join(data_dir, 'negative_train.csv'), index=False)
negative_val.to_csv(os.path.join(data_dir, 'negative_val.csv'), index=False)
negative_test.to_csv(os.path.join(data_dir, 'negative_test.csv'), index=False)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Process and save the datasets for training, validation, and testing
train_df = process_dataset(train_data, train_data.groupby('CustomerID'), negative_train, "train")
val_df = process_dataset(validation_data, validation_data.groupby('CustomerID'), negative_val, "validation")
test_df = process_dataset(test_data, test_data.groupby('CustomerID'), negative_test, "test")

train_df.to_csv(os.path.join(data_dir, 'trainForBERT_WCP.csv'), index=False)
val_df.to_csv(os.path.join(data_dir, 'valForBERT_WCP.csv'), index=False)
test_df.to_csv(os.path.join(data_dir, 'testForBERT_WCP.csv'), index=False)

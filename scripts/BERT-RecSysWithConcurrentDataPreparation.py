import os
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import GroupShuffleSplit
from transformers import BertTokenizer

# Function Definitions

def split_user_data(group_data):
    """Split group data into prompts and a positive example."""
    user_prompt = group_data.iloc[:-1].sort_values(by='InvoiceDate')
    user_positive = group_data.iloc[-1]
    return user_prompt, user_positive

def format_next_purchase(user_text):
    """Format text for next purchase prediction."""
    item_text = f'## next purchase prediction\n{user_text.Description}'
    return item_text

def get_longer_text(text_1, text_2):
    """Determine the longer text between two given texts."""
    text_max = text_1 if len(tokenizer(text_1, padding=True, truncation=False).input_ids) > len(tokenizer(text_2, padding=True, truncation=False).input_ids) else text_2
    return text_max

def add_text_if_fits(current_text, addition, tokenizer, item_max):
    """
    Adds additional text to the given text and checks if the result fits within the token limit.
    If it does not exceed the limit, it appends the additional text to the current text;
    if it exceeds, it returns None.
    """
    # Calculate the number of tokens for the combined current and additional text using the tokenizer
    new_text = current_text + addition
    if len(tokenizer(new_text, padding=True, truncation=False).input_ids) < 512:
        return new_text  # Return the updated text if within the limit
    else:
        return None  # Return None if the limit is exceeded

def compile_order_history(user_prompt, tokenizer, item_max):
    # Initialize the order history text with a heading.
    user_text = "## order history"
    # Group the input data by invoice date to compile orders day by day.
    user_prompt = user_prompt.groupby('InvoiceDate')
    
    for date, group in user_prompt:
        # For each date, start compiling order details.
        date_text = f'\nOrder details for {date}'
        # Attempt to add the date text, checking if it fits within the token limit.
        new_user_text = add_text_if_fits(user_text, date_text, tokenizer, item_max)
        if not new_user_text:  # If adding the text would exceed the token limit,
            return user_text  # return the current state of the compiled text.
        else:
            user_text = new_user_text  # Otherwise, update the compiled text.
        
        # If there are multiple items in an order, add a concurrent purchase marker.
        if len(group) > 1:
            concurrent_purchase_text = ' [Concurrent Purchase]'
            new_user_text = add_text_if_fits(user_text, concurrent_purchase_text, tokenizer, item_max)
            if not new_user_text:  # Check again for token limit exceedance.
                return user_text
            else:
                user_text = new_user_text
        
        # Separate the order details with a colon.
        user_text = add_text_if_fits(user_text, ': ', tokenizer, item_max)
        if not user_text: return user_text  # Check for token limit before adding descriptions.
        
        # For each item description in the order,
        for description in group['Description']:
            description_text = description + ', '
            # Attempt to add item descriptions, checking each time for token limits.
            new_user_text = add_text_if_fits(user_text, description_text, tokenizer, item_max)
            if not new_user_text:  # If limit exceeded,
                return user_text  # return the text as-is.
            else:
                user_text = new_user_text  # Otherwise, update with the new item.
                
    return user_text  # Return the fully compiled order history text.

def negative_sampling(df, unique_product_ids, user_id, N):
    """Perform negative sampling for a given user."""
    purchased_product_ids = set(df[df['CustomerID'] == user_id]['CustomerID'])
    negative_product_ids = unique_product_ids - purchased_product_ids
    sampled_product_ids = np.random.choice(list(negative_product_ids), N, replace=False)
    return sampled_product_ids

def create_negative_samples_dataframe(df, unique_user_ids, unique_product_ids, N):
    """Create a DataFrame of negative samples."""
    negative_samples = []
    for user_id in unique_user_ids:
        sampled_product_ids = negative_sampling(df, unique_product_ids, user_id, N)
        user_negative_samples = [{'CustomerID': user_id, 'Description': pid} for pid in sampled_product_ids]
        negative_samples.extend(user_negative_samples)
    negative_product_id_df = pd.DataFrame(negative_samples)
    return negative_product_id_df

def generate_dataset(group_keys, grouped_data, negative_data, mode):
    """Generate datasets for model training or evaluation."""
    sentence_prev, sentence_next, labels = [], [], []
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
            
            for i in range(50):
                sentence_prev.append(user_text)
                
    return sentence_prev, sentence_next, labels

# Main Code

# Fetch and preprocess the dataset
data = fetch_ucirepo(id=352).data.features
data = data.copy()
data['total_order_cnt'] = data.groupby('CustomerID')['Description'].transform('count')
data = data[data['total_order_cnt'] > 1].drop(['total_order_cnt'], axis=1).reset_index(drop=True)

# Split data into training, validation, and testing sets
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_val_idx, test_idx = next(gss.split(data, groups=data['CustomerID']))
train_val_data, test_data = data.iloc[train_val_idx], data.iloc[test_idx]

gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
train_idx, val_idx = next(gss.split(train_val_data, groups=train_val_data['CustomerID']))
train_data, validation_data = train_val_data.iloc[train_idx], train_val_data.iloc[val_idx]

# Ensure data directory exists
data_dir = "data/BERT_ConcurrentPurchases"
os.makedirs(data_dir, exist_ok=True)

# Save datasets to CSV
train_data.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
validation_data.to_csv(os.path.join(data_dir, 'validation_data.csv'), index=False)
test_data.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Generate and save datasets for BERT
for mode, dataset, grouped_data in [("train", train_data, grouped_train), ("validation", validation_data, grouped_val), ("test", test_data, grouped_test)]:
    group_keys = grouped_data.groups.keys()
    sentence_prev, sentence_next, labels = generate_dataset(group_keys, grouped_data, negative_train if mode == "train" else (negative_val if mode == "validation" else negative_test), mode)
    
    df = pd.DataFrame({'prev': sentence_prev, 'next': sentence_next, 'label': labels})
    df.to_csv(os.path.join(data_dir, f'{mode}ForBERT_WCP.csv'), index=False)

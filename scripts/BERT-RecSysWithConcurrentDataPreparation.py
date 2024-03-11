import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from transformers import BertTokenizer
from ucimlrepo import fetch_ucirepo

# Function Definitions

# Define a function to perform something similar to negative sampling for each user_id,
# sampling N product_ids not purchased by the user from the set of unique product_ids.
def negative_sampling(df, unique_product_ids, user_id, N):
    purchased_product_ids = set(df[df['CustomerID'] == user_id]['CustomerID'])
    negative_product_ids = unique_product_ids - purchased_product_ids
    sampled_product_ids = np.random.choice(list(negative_product_ids), N, replace=False)
    return sampled_product_ids

# Adjusted function to create a DataFrame of negative samples.
def create_negative_samples_dataframe(df, unique_user_ids, unique_product_ids, N):
    negative_samples = []
    for user_id in unique_user_ids:
        sampled_product_ids = negative_sampling(df, unique_product_ids, user_id, N)
        user_negative_samples = [{'CustomerID': user_id, 'Description': pid} for pid in sampled_product_ids]
        negative_samples.extend(user_negative_samples)
    negative_product_id_df = pd.DataFrame(negative_samples)
    return negative_product_id_df

# Splits user data into prompts and positive examples.
def split_user_data(group_data):
    user_prompt = group_data.iloc[:-1].sort_values(by='InvoiceDate')
    user_positive = group_data.iloc[-1]
    return user_prompt, user_positive

# Formats the next purchase prediction, ensuring correct string output.
def format_next_purchase(user_text):
    try:
        item_description = user_text.Description.iloc[0]
    except IndexError:
        item_description = user_text.Description
    except AttributeError:
        item_description = user_text.Description
    item_text = f'## next purchase prediction\n{item_description}'
    return item_text

# Determines the longer text between two given texts based on token count.
def get_longer_text(text_1, text_2):
    text_max = text_1 if len(tokenizer(text_1, padding=True, truncation=False).input_ids) > len(tokenizer(text_2, padding=True, truncation=False).input_ids) else text_2
    return text_max

# Attempts to add additional text to the given text without exceeding the token limit.
def add_text_if_fits(current_text, addition, tokenizer, item_max):
    new_text = current_text + addition
    if len(tokenizer(new_text, item_max, padding=True, truncation=False).input_ids) < 512:
        return new_text
    else:
        return None

# Compiles order history by grouping input data by invoice date and organizing orders chronologically.
def compile_order_history(user_prompt, tokenizer, item_max):
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

# Generates the dataset for BERT model training, validation, or testing.
def generate_dataset(group_keys, grouped_data, negative_data, mode):
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

# Main Data Processing

# Fetch and preprocess data
data = fetch_ucirepo(id=352)
data = data.data.features

# Filter out users with only one purchase
data['total_order_cnt'] = data.groupby('CustomerID')['Description'].transform('count')
data = data[data['total_order_cnt'] > 1]
data.drop(['total_order_cnt'], axis=1, inplace=True)
data.reset_index(drop=True, inplace=True)

# Split data into train, validation, and test sets
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_val_idx, test_idx = next(gss.split(data, groups=data['CustomerID']))
train_val_data = data.iloc[train_val_idx]
test_data = data.iloc[test_idx]

gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
train_idx, val_idx = next(gss.split(train_val_data, groups=train_val_data['CustomerID']))
train_data = train_val_data.iloc[train_idx]
validation_data = train_val_data.iloc[val_idx]

# Ensure data directory exists and save data splits
data_dir = "data/BERT_ConcurrentPurchases"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
train_data.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
validation_data.to_csv(os.path.join(data_dir, 'validation_data.csv'), index=False)
test_data.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)

# Generate negative sample DataFrame for training data
# Creates negative samples for each user by selecting products not purchased by them
unique_user_ids = train_data['CustomerID'].unique()
unique_product_ids = set(train_data['Description'].unique())

negative_product_id_df = create_negative_samples_dataframe(train_data, unique_user_ids, unique_product_ids, N=1)

# Merge negative samples with last purchase info and clean up columns
# This step merges the negative samples with product descriptions and adjusts the CustomerID column
xx = train_data.groupby('Description', as_index=False).last()
negative_train = pd.merge(negative_product_id_df, xx, on='Description', how='left').drop(['CustomerID_y'], axis=1).rename(columns={'CustomerID_x': 'CustomerID'})

# Repeat negative sampling process for validation data
# Similar to training data, generates negative samples for validation set
unique_user_ids = validation_data['CustomerID'].unique()
unique_product_ids = set(validation_data['Description'].unique())
negative_product_id_df = create_negative_samples_dataframe(validation_data, unique_user_ids, unique_product_ids, N=1)
xx = validation_data.groupby('Description', as_index=False).last()
negative_val = pd.merge(negative_product_id_df, xx, on='Description', how='left').drop(['CustomerID_y'], axis=1).rename(columns={'CustomerID_x': 'CustomerID'})

# Perform negative sampling for test data with increased sample count
# Generates 49 negative samples for each user in the test set to evaluate model's performance
unique_user_ids = test_data['CustomerID'].unique()
unique_product_ids = set(test_data['Description'].unique())
negative_product_id_df = create_negative_samples_dataframe(test_data, unique_user_ids, unique_product_ids, N=49)
xx = test_data.groupby('Description', as_index=False).last()
negative_test = pd.merge(negative_product_id_df, xx, on='Description', how='left').drop(['CustomerID_y'], axis=1).rename(columns={'CustomerID_x': 'CustomerID'})

# Export negative sample DataFrames to CSV files
# Saves the negative samples for training, validation, and test sets to the specified directory
negative_train.to_csv(os.path.join(data_dir, 'negative_train.csv'), index=False)
negative_val.to_csv(os.path.join(data_dir, 'negative_val.csv'), index=False)
negative_test.to_csv(os.path.join(data_dir, 'negative_test.csv'), index=False)

# Prepare data for BERT model
# Tokenizes and formats data for input into BERT model, generating sequences of user's purchase history and next product prediction
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Process training data
# Groups training data by CustomerID and generates sequences for model training
grouped_train = train_data.groupby('CustomerID')
group_keys = grouped_train.groups.keys()
sentence_prev, sentence_next, labels = generate_dataset(group_keys, grouped_train, negative_train, "train")

# Prepare DataFrame for training data
train = pd.DataFrame(columns = ['prev', 'next', 'label'])
train['prev'] = sentence_prev
train['next'] = sentence_next
train['label'] = labels

# Process validation data
# Similar processing for validation data to create input sequences and labels
grouped_val = validation_data.groupby('CustomerID')
group_keys = grouped_val.groups.keys()
sentence_prev, sentence_next, labels = generate_dataset(group_keys, grouped_val, negative_val, "validation")
val = pd.DataFrame(columns = ['prev', 'next', 'label'])
val['prev'] = sentence_prev
val['next'] = sentence_next
val['label'] = labels

# Process test data
# Generates sequences for test data, including negative samples for comprehensive evaluation
grouped_test = test_data.groupby('CustomerID')
group_keys = grouped_test.groups.keys()
sentence_prev, sentence_next, labels = generate_dataset(group_keys, grouped_test, negative_test, "test")
test = pd.DataFrame(columns = ['prev', 'next', 'label'])
test['prev'] = sentence_prev
test['next'] = sentence_next
test['label'] = labels

# Filter out sequences that only contain the order history heading
train = train[train.prev != '## order history']
val = val[val.prev != '## order history']
test = test[test.prev != '## order history']

# Save processed datasets to CSV files
# Outputs the final processed datasets for training, validation, and test sets to the specified directory for model training
train.to_csv(os.path.join(data_dir, 'trainForBERT_WCP.csv'), index=False)
val.to_csv(os.path.join(data_dir, 'valForBERT_WCP.csv'), index=False)
test.to_csv(os.path.join(data_dir, 'testForBERT_WCP.csv'), index=False)

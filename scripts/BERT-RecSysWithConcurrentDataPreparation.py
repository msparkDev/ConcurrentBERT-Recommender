import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from transformers import BertTokenizer
from ucimlrepo import fetch_ucirepo


# Function Definitions

# Define a function to sample N non-purchased products for each user as negative examples.
def negative_sampling(df, unique_product_ids, user_id, N):
    purchased_product_ids = set(df[df['CustomerID'] == user_id]['CustomerID'])
    negative_product_ids = unique_product_ids - purchased_product_ids
    sampled_product_ids = np.random.choice(list(negative_product_ids), N, replace=False)
    return sampled_product_ids


# Create a DataFrame containing negative samples for each user.
def create_negative_samples_dataframe(df, unique_user_ids, unique_product_ids, N):
    negative_samples = []
    for user_id in unique_user_ids:
        sampled_product_ids = negative_sampling(df, unique_product_ids, user_id, N)
        user_negative_samples = [{'CustomerID': user_id, 'Description': pid} for pid in sampled_product_ids]
        negative_samples.extend(user_negative_samples)
    negative_product_id_df = pd.DataFrame(negative_samples)
    return negative_product_id_df


# Split user data into historical purchases (prompts) and the last purchase (positive example).
def split_user_data(group_data):
    user_prompt = group_data.iloc[:-1].sort_values(by='InvoiceDate')
    user_positive = group_data.iloc[-1]
    return user_prompt, user_positive


# Format the product description for prediction, handling potential issues.
def format_next_purchase(user_text):
    try:
        item_description = user_text.Description.iloc[0]
    except IndexError:
        item_description = user_text.Description
    except AttributeError:
        item_description = user_text.Description
    item_text = f'## next purchase prediction\n{item_description}'
    return item_text


# Determine the text with more tokens between two options.
def get_longer_text(text_1, text_2):
    text_max = text_1 if len(tokenizer(text_1, padding=True, truncation=False).input_ids) > len(
        tokenizer(text_2, padding=True, truncation=False).input_ids) else text_2
    return text_max


# Try adding additional text without exceeding the maximum token limit.
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
        ctemp = temp + f': {group["Description"].iloc[0]}'
        new_user_text = add_text_if_fits(user_text, ctemp, tokenizer, item_max)
        if new_user_text is not None:
            user_text += (temp + ': ')
        else:
            return user_text

        for description in group['Description']:
            temp = description + ', '
            new_user_text = add_text_if_fits(user_text, temp, tokenizer, item_max)
            if new_user_text is not None:
                user_text = new_user_text
            else:
                return user_text
    return user_text

# Generate datasets for BERT model training, validation, or testing.
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

            for i in range(len(user_negative) + 1):
                sentence_prev.append(user_text)

    return sentence_prev, sentence_next, labels


# Prepare negative samples by merging product descriptions.
def prepare_negative_samples(data, unique_user_ids, N):
    unique_product_ids = set(data['Description'].unique())
    negative_product_id_df = create_negative_samples_dataframe(data, unique_user_ids, unique_product_ids, N)
    xx = data.groupby('Description', as_index=False).last()
    negative_samples_merged = pd.merge(negative_product_id_df, xx, on='Description', how='left')
    negative_samples_merged.drop(['CustomerID_y'], axis=1, inplace=True)
    negative_samples_merged.rename(columns={'CustomerID_x': 'CustomerID'}, inplace=True)
    return negative_samples_merged


# Process datasets for training, validation, or testing, creating sequences and labels.
def process_dataset(data, grouped_data, negative_data, mode):
    group_keys = grouped_data.groups.keys()
    sentence_prev, sentence_next, labels = generate_dataset(group_keys, grouped_data, negative_data, mode)
    df = pd.DataFrame({
        'prev': sentence_prev,
        'next': sentence_next,
        'label': labels
    })
    return df


# Main Data Processing

# Fetch and preprocess data
data = fetch_ucirepo(id=352)
data = data.data.features

# Filter and prepare data
data['total_order_cnt'] = data.groupby('CustomerID')['Description'].transform('count')
data = data[data['total_order_cnt'] > 1]
data.drop(['total_order_cnt'], axis=1, inplace=True)
data.reset_index(drop=True, inplace=True)

# Split data into sets
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_val_idx, test_idx = next(gss.split(data, groups=data['CustomerID']))
train_val_data = data.iloc[train_val_idx]
test_data = data.iloc[test_idx]

# Further split into training and validation sets
gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
train_idx, val_idx = next(gss.split(train_val_data, groups=train_val_data['CustomerID']))
train_data = train_val_data.iloc[train_idx]
validation_data = train_val_data.iloc[val_idx]

# Prepare directories and save data
data_dir = "data/BERT_ConcurrentPurchases"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
train_data.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
validation_data.to_csv(os.path.join(data_dir, 'validation_data.csv'), index=False)
test_data.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)

# Prepare negative samples for all sets
negative_train = prepare_negative_samples(train_data, train_data['CustomerID'].unique(), N=1)
negative_val = prepare_negative_samples(validation_data, validation_data['CustomerID'].unique(), N=1)
negative_test = prepare_negative_samples(test_data, test_data['CustomerID'].unique(), N=49)

# Save negative samples to CSV
negative_train.to_csv(os.path.join(data_dir, 'negative_train.csv'), index=False)
negative_val.to_csv(os.path.join(data_dir, 'negative_val.csv'), index=False)
negative_test.to_csv(os.path.join(data_dir, 'negative_test.csv'), index=False)

# Initialize tokenizer and process datasets for BERT
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased', do_lower_case=False)

grouped_train = train_data.groupby('CustomerID')
grouped_val = validation_data.groupby('CustomerID')
grouped_test = test_data.groupby('CustomerID')

train = process_dataset(train_data, grouped_train, negative_train, "train")
val = process_dataset(validation_data, grouped_val, negative_val, "validation")
test = process_dataset(test_data, grouped_test, negative_test, "test")

# Finalize and save processed datasets for model training
train.to_csv(os.path.join(data_dir, 'trainForBERT_WCP.csv'), index=False)
val.to_csv(os.path.join(data_dir, 'valForBERT_WCP.csv'), index=False)
test.to_csv(os.path.join(data_dir, 'testForBERT_WCP.csv'), index=False)

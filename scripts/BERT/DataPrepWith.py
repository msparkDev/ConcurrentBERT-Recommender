import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from transformers import BertTokenizer

# Function to perform negative sampling for each user_id,
# sampling N product_ids not purchased by the user from the set of unique product_ids.
def negative_sampling(df, unique_product_ids, user_id, N):
    """
    Performs negative sampling for a given user.

    Parameters:
    - df: DataFrame containing the purchase history.
    - unique_product_ids: Set of all unique product IDs.
    - user_id: The ID of the user for whom negative samples are to be generated.
    - N: Number of negative samples to generate.

    Returns:
    - sampled_product_ids: N randomly chosen product IDs not purchased by the user.
    """
    purchased_product_ids = set(df[df['user_id'] == user_id]['user_id'])
    negative_product_ids = unique_product_ids - purchased_product_ids
    sampled_product_ids = np.random.choice(list(negative_product_ids), N, replace=False)
    return sampled_product_ids


# Function to create a DataFrame of negative samples.
def create_negative_samples_dataframe(df, unique_user_ids, unique_product_ids, N):
    """
    Creates a DataFrame of negative samples for each user.

    Parameters:
    - df: DataFrame containing the purchase history.
    - unique_user_ids: List or array of unique user IDs.
    - unique_product_ids: Set of all unique product IDs.
    - N: Number of negative samples to generate per user.

    Returns:
    - negative_product_id_df: DataFrame containing negative samples.
    """
    negative_samples = []
    for user_id in unique_user_ids:
        sampled_product_ids = negative_sampling(df, unique_product_ids, user_id, N)
        user_negative_samples = [{'CustomerID': user_id, 'product_id': pid} for pid in sampled_product_ids]
        negative_samples.extend(user_negative_samples)
    negative_product_id_df = pd.DataFrame(negative_samples)
    return negative_product_id_df


# Function to split user data into prompts and positive examples.
def split_user_data(group_data):
    """
    Splits user data into prompts and a single positive example.

    Parameters:
    - group_data: DataFrame of grouped data for a single user.

    Returns:
    - user_prompt: DataFrame containing all but the last purchase of the user.
    - user_positive: Series containing the last purchase of the user.
    """
    user_prompt = group_data.iloc[:-1].sort_values(by='initial_paid_at')
    user_positive = group_data.iloc[-1]
    return user_prompt, user_positive

# Function to format the next purchase prediction for output.
def format_next_purchase(user_text):
    """
    Formats the text for the next purchase prediction.

    Parameters:
    - user_text: DataFrame or Series containing the next purchase's description.

    Returns:
    - item_text: Formatted string for the next purchase prediction.
    """
    try:
        if pd.notna(user_text.attribute_values.iloc[0]):
            item_description = f'{user_text.category_name.iloc[0]} - {user_text.product_name.iloc[0]} {user_text.attribute_values.iloc[0]}'
        else:
            item_description = f'{user_text.category_name.iloc[0]} - {user_text.product_name.iloc[0]}'

    except IndexError:
        if pd.notna(user_text.attribute_values):
            item_description = f'{user_text.category_name} - {user_text.product_name} {user_text.attribute_values}'
        else:
            item_description = f'{user_text.category_name} - {user_text.product_name}'
    except AttributeError:
        if pd.notna(user_text.attribute_values):
            item_description = f'{user_text.category_name} - {user_text.product_name} {user_text.attribute_values}'
        else:
            item_description = f'{user_text.category_name} - {user_text.product_name}'

    item_text = f'## 다음 구매 상품 예측\n{item_description}'
    return item_text


# Function to determine the longer text between two given texts based on token count.
def get_longer_text(text_1, text_2, tokenizer):
    """
    Determines the longer text between two texts based on token count.

    Parameters:
    - text_1: First text to compare.
    - text_2: Second text to compare.
    - tokenizer: Tokenizer object used for tokenizing the texts.

    Returns:
    - text_max: The longer text of the two inputs.
    """
    text_max = text_1 if len(tokenizer(text_1, padding=True, truncation=False).input_ids) > \
                         len(tokenizer(text_2, padding=True, truncation=False).input_ids) else text_2
    return text_max


# Function to attempt adding additional text without exceeding token limit.
def add_text_if_fits(current_text, addition, tokenizer, item_max):
    """
    Attempts to add additional text to the given text without exceeding the token limit.

    Parameters:
    - current_text: The current text string to which addition is attempted.
    - addition: The text to add to the current text.
    - tokenizer: Tokenizer object used for tokenizing the texts.
    - item_max: Maximum item size for consideration in tokenization.

    Returns:
    - new_text or None: The new text string with the addition or None if exceeding token limit.
    """
    new_text = current_text + addition
    if len(tokenizer(new_text, item_max, padding=True, truncation=False).input_ids) < 512:
        return new_text
    else:
        return None


# Function to compile user's order history.
def compile_order_history(user_prompt, tokenizer, item_max):
    """
    Compiles the order history for a user into a text string.

    Parameters:
    - user_prompt: DataFrame containing the user's orders except the last.
    - tokenizer: Tokenizer object used for tokenizing the texts.
    - item_max: Maximum item size for consideration in tokenization.

    Returns:
    - user_text: Compiled order history as a text string.
    """
    user_text = "## 주문 히스토리"
    user_prompt_grouped = user_prompt.groupby('initial_paid_at', sort=False)

    ctemp = ''
    for date, group in user_prompt_grouped:
        temp = f'\n{date}의 주문 내역'
        if len(group) > 1:
            ctemp += temp + ' [동시구매]'
        if pd.notna(group.iloc[0].attribute_values):
            ctemp = ctemp + f': {group["category_name"].iloc[0]} - {group["product_name"].iloc[0]} {group["attribute_values"].iloc[0]}'
        else:
            ctemp = ctemp + f': {group["category_name"].iloc[0]} - {group["product_name"].iloc[0]}'

        new_user_text = add_text_if_fits(user_text, ctemp, tokenizer, item_max)
        if new_user_text is not None:
            temp = (ctemp + ': ')
        else:
            user_text += temp
            print(user_text)
            return user_text

        len_group = len(group)
        count = 0
        for index, row in group.iterrows():
            if count > 0:
                ctemp += ', '
            else:
                ctemp += ''
            count += 1
            if pd.notna(row.attribute_values):
                ctemp += f'{row["category_name"]} - {row["product_name"]} {row["attribute_values"]}'
            else:
                ctemp += f'{row["category_name"]} - {row["product_name"]}'

            new_user_text = add_text_if_fits(user_text, ctemp, tokenizer, item_max)
            if new_user_text is not None:
                temp = ctemp
                continue
            else:
                user_text += temp
                return user_text

    return user_text


# Function to generate datasets for BERT model training, validation, or testing.
def generate_dataset(group_keys, grouped_data, negative_data, mode):
    """
    Generates the dataset for training, validation, or testing the BERT model.

    Parameters:
    - group_keys: Keys identifying unique groups in the data, typically unique user IDs.
    - grouped_data: DataFrame grouped by user or another entity.
    - negative_data: DataFrame containing negative samples.
    - mode: A string indicating the mode of dataset generation ('train', 'validation', or 'test').

    Returns:
    - A tuple of lists: sentence_prev (list of user's order history texts),
      sentence_next (list of next purchase or negative sample texts),
      and labels (list indicating whether the sentence_next is a positive (1) or negative (0) example).
    """
    sentence_prev = []
    sentence_next = []
    labels = []

    for key in group_keys:
        group_data = grouped_data.get_group(key).copy()
        user_prompt, user_positive = split_user_data(group_data)
        user_negative = negative_data[negative_data.user_id == key]

        item_pos = format_next_purchase(user_positive)
        sentence_next.append(item_pos)
        labels.append(1)

        if mode in ["train", "validation"]:
            item_neg = format_next_purchase(user_negative)
            sentence_next.append(item_neg)
            labels.append(0)

            item_max = get_longer_text(item_pos, item_neg, tokenizer)

            user_text = compile_order_history(user_prompt, tokenizer, item_max)

            for i in range(2):
                sentence_prev.append(user_text)

        elif mode == "test":
            item_max = item_pos  # In test mode, only positive items are considered initially.

            for i, (index, neg) in enumerate(user_negative.iterrows()):
                item_neg = format_next_purchase(neg)
                sentence_next.append(item_neg)
                labels.append(0)

                item_max = get_longer_text(item_max, item_neg, tokenizer)

            user_text = compile_order_history(user_prompt, tokenizer, item_max)

            for i in range(len(user_negative) + 1):
                sentence_prev.append(user_text)

    return sentence_prev, sentence_next, labels


# Function to prepare negative samples from the dataset.
def prepare_negative_samples(data, unique_user_ids, N):
    """
    Prepares negative samples for each user in the dataset.

    Parameters:
    - data: DataFrame containing the user purchase history.
    - unique_user_ids: Array of unique user IDs.
    - N: Number of negative samples to generate for each user.

    Returns:
    - negative_samples_merged: DataFrame containing negative samples with merged product descriptions.
    """
    # Generate unique product IDs from the data
    unique_product_ids = set(data['product_id'].unique())

    # Create negative samples dataframe
    negative_product_id_df = create_negative_samples_dataframe(data, unique_user_ids, unique_product_ids, N)

    # Merge negative samples with product descriptions from the last purchase
    xx = data.groupby('product_id', as_index=False).last()
    negative_samples_merged = pd.merge(negative_product_id_df, xx, on='product_id', how='left')
    negative_samples_merged.drop(['user_id_y'], axis=1, inplace=True)
    negative_samples_merged.rename(columns={'user_id_x': 'user_id'}, inplace=True)

    return negative_samples_merged


# Function to process and format the dataset for BERT model input.
def process_dataset(data, grouped_data, negative_data, mode):
    """
    Processes the dataset for training, validation, or testing and formats it for BERT model input.

    Parameters:
    - data: DataFrame containing the user purchase history.
    - grouped_data: DataFrame grouped by unique identifiers (e.g., CustomerID).
    - negative_data: DataFrame containing negative samples for each user.
    - mode: String indicating the mode ('train', 'validation', 'test') of dataset processing.

    Returns:
    - df: DataFrame containing processed and formatted data for BERT model input.
    """
    # Get the unique keys for groups, typically CustomerIDs.
    group_keys = grouped_data.groups.keys()

    # Generate sequences for training, validation, or testing.
    sentence_prev, sentence_next, labels = generate_dataset(group_keys, grouped_data, negative_data, mode)

    # Create a DataFrame from the generated sequences and labels.
    df = pd.DataFrame({
        'prev': sentence_prev,
        'next': sentence_next,
        'label': labels
    })

    return df

# Main Data Processing

# Fetch and preprocess data from UCI Machine Learning Repository
data = pd.read_csv("data/raw_data_katchers.csv")

# Step 1: Filter users with more than one unique purchase date to ensure meaningful historical data
data['unique_purchase_dates'] = data.groupby('user_id')['initial_paid_at'].transform(lambda x: x.nunique())
data = data[data['unique_purchase_dates'] > 1].drop(['unique_purchase_dates'], axis=1)

# Step 2: Identify the most recent purchase date for each user
df_grouped = data.groupby('user_id')['initial_paid_at'].max().reset_index(name='initial_paid_at')

# Step 3: Merge to isolate the transactions occurring on the latest purchase date
df_max_all = data.merge(df_grouped, how='inner', on=['user_id', 'initial_paid_at'])

# Step 4: Randomly select one transaction per user from their latest purchase date
df_max = df_max_all.groupby('user_id').sample(n=1, random_state=42).reset_index(drop=True)

# Step 5: Create a dataset excluding the last purchases (for training on historical data)
df_joined = data.merge(df_grouped, how='inner', on='user_id', suffixes=['', '_y'])
df_joined['initial_paid_at_mismatch'] = df_joined['initial_paid_at'] != df_joined['initial_paid_at_y']
df_not_max = df_joined[df_joined['initial_paid_at_mismatch']].drop(['initial_paid_at_y', 'initial_paid_at_mismatch'], axis=1)

# Step 6: Combine datasets to form a complete dataset excluding only the selected final purchases
df_processed = pd.concat([df_max, df_not_max]).reset_index(drop=True)

# Step 7: Further filter users to ensure they have a meaningful number of transactions
df_processed['total_order_cnt'] = df_processed.groupby('user_id')['product_id'].transform('count')
df_processed = df_processed[(df_processed['total_order_cnt'] > 1) & (df_processed['total_order_cnt'] < 68)]
df_processed = df_processed.drop(['total_order_cnt'], axis=1).reset_index(drop=True)

# Step 8: Split the data into training, validation, and testing sets
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_val_idx, test_idx = next(gss.split(df_processed, groups=df_processed['user_id']))
train_val_data, test_data = df_processed.iloc[train_val_idx], df_processed.iloc[test_idx]

gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
train_idx, val_idx = next(gss.split(train_val_data, groups=train_val_data['user_id']))
train_data, val_data = train_val_data.iloc[train_idx], train_val_data.iloc[val_idx]

# Step 9: Save the split data sets into designated directories
data_dir = "data/BERT/Concurrent"
os.makedirs(data_dir, exist_ok=True)
train_data.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
val_data.to_csv(os.path.join(data_dir, 'val_data.csv'), index=False)
test_data.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)

# Step 10: Prepare negative samples for the BERT model
negative_train = prepare_negative_samples(pd.concat([df_max_all[df_max_all.CustomerID.isin(train_data['user_id'].unique())], train_data]), train_data['user_id'].unique(), N=1)
negative_val = prepare_negative_samples(pd.concat([df_max_all[df_max_all.CustomerID.isin(val_data['user_id'].unique())], val_data]), val_data['user_id'].unique(), N=1)
negative_test = prepare_negative_samples(pd.concat([df_max_all[df_max_all.CustomerID.isin(test_data['user_id'].unique())], test_data]), test_data['user_id'].unique(), N=49)

# Step 11: Save the prepared negative samples
negative_train.to_csv(os.path.join(data_dir, 'negative_train.csv'), index=False)
negative_val.to_csv(os.path.join(data_dir, 'negative_val.csv'), index=False)
negative_test.to_csv(os.path.join(data_dir, 'negative_test.csv'), index=False)

# Step 12: Tokenize and format data for the BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
grouped_train, grouped_val, grouped_test = train_data.groupby('user_id'), val_data.groupby('user_id'), test_data.groupby('user_id')

# Step 13: Process datasets for the BERT model
train_dataset = process_dataset(train_data, grouped_train, negative_train, "train")
val_dataset = process_dataset(val_data, grouped_val, negative_val, "validation")
test_dataset = process_dataset(test_data, grouped_test, negative_test, "test")

# Step 14: Save the final processed datasets for model training
train_dataset.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
val_dataset.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
test_dataset.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

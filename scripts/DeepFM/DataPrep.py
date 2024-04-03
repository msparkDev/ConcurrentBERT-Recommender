import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from transformers import BertTokenizer
from scripts.BERT.DataPrepWith import split_user_data

def process_dataset(group_keys, grouped_data, negative_data, mode):
    """
    Processes the dataset to format it for use in recommendation system models, 
    focusing on both concurrent and singular purchase patterns.
    
    Parameters:
    - group_keys: The unique keys identifying groups in the dataset.
    - grouped_data: DataFrame grouped by CustomerID.
    - negative_data: DataFrame containing negative samples for each customer.
    - mode: Operation mode ('train', 'validation', 'test') indicating how the dataset should be processed.
    
    Returns:
    - A DataFrame structured for input into the recommendation model.
    """
    # Define column names dynamically based on the requirements for concurrent purchase tracking
    columns = [f'{i+1}th_purchase_product_id' for i in range(5)] + [f'{i+1}th_concurrent_purchase_flag' for i in range(5)] + ['6th_purchase_product_id', 'label']
    data_rows = []

    for key in group_keys:
        group_data = grouped_data.get_group(key).copy()
        user_prompt, user_positive = split_user_data(group_data)  # Split data into prompt and positive examples
        
        # Extract last 5 purchase records for each user
        last_5_purchases = user_prompt.sort_values(by='InvoiceDate', ascending=False).head(5)
        
        # Generate and append a positive example row
        new_row_positive = create_new_row(last_5_purchases, user_positive)
        data_rows.append(new_row_positive)
        
        if mode in ["train", "validation"]:
            # For training/validation, use a single negative example per user
            user_negative_sample = negative_data[negative_data.CustomerID == key]
            new_row_negative = create_new_row(last_5_purchases, user_negative_sample.iloc[0], label=0)
            data_rows.append(new_row_negative)
        elif mode == "test":
            # For testing, use all available negative examples
            user_negative_samples = negative_data[negative_data.CustomerID == key]
            for _, user_negative in user_negative_samples.iterrows():
                new_row_negative = create_new_row(last_5_purchases, user_negative, label=0)
                data_rows.append(new_row_negative)

    data = pd.DataFrame(data_rows, columns=columns).fillna(0)
    return data

def create_new_row(last_5_purchases, next_purchase, label=1):
    """
    Creates a new row for the dataset representing a user's purchase history and next purchase.
    
    Parameters:
    - last_5_purchases: DataFrame containing the last five purchases of the user.
    - next_purchase: The next purchase (positive or negative example).
    - label: Indicates whether the example is positive (1) or negative (0).
    
    Returns:
    - A dictionary representing the new row to be added to the dataset.
    """
    new_row = {}
    
    # Populate the new row with product IDs and concurrent purchase flags
    for i in range(len(last_5_purchases)):
        new_row[f'{i+1}th_purchase_product_id'] = last_5_purchases.iloc[i].StockCode
        is_concurrent = 0
        if i > 0 and last_5_purchases.iloc[i].InvoiceDate == last_5_purchases.iloc[i-1].InvoiceDate or \
           i < len(last_5_purchases) - 1 and last_5_purchases.iloc[i].InvoiceDate == last_5_purchases.iloc[i+1].InvoiceDate:
            is_concurrent = 1
        new_row[f'{i+1}th_concurrent_purchase_flag'] = is_concurrent
    
    # Add the next purchase and label
    new_row['6th_purchase_product_id'] = next_purchase.StockCode if 'StockCode' in dir(next_purchase) else 0
    new_row['label'] = label
    
    return new_row

# Main data processing flow
data_dir = "data/BERT/Concurrent"  # Original data directory
data_dir_1 = "data/DeepFM/Concurrent"  # New data directory for processed files
data_dir_2 = "data/DeepFM/Single"  # New data directory for processed files

# Ensure the new data directory exists
os.makedirs(data_dir_1, exist_ok=True)
os.makedirs(data_dir_2, exist_ok=True)

# Load data and negative samples
train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
val_data = pd.read_csv(os.path.join(data_dir, 'val_data.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
negative_train = pd.read_csv(os.path.join(data_dir, 'negative_train.csv'))
negative_val = pd.read_csv(os.path.join(data_dir, 'negative_val.csv'))
negative_test = pd.read_csv(os.path.join(data_dir, 'negative_test.csv'))

# Process datasets for each mode
train_df = process_dataset(train_data.groupby('CustomerID').groups, train_data.groupby('CustomerID'), negative_train, "train")
val_df = process_dataset(val_data.groupby('CustomerID').groups, val_data.groupby('CustomerID'), negative_val, "validation")
test_df = process_dataset(test_data.groupby('CustomerID').groups, test_data.groupby('CustomerID'), negative_test, "test")

# Drop 'concurrent_purchase' columns for models not utilizing concurrent purchase data
columns_to_drop = [col for col in train_df.columns if 'concurrent_purchase' in col]
train_df_dropped = train_df.drop(columns=columns_to_drop)
val_df_dropped = val_df.drop(columns=columns_to_drop)
test_df_dropped = test_df.drop(columns=columns_to_drop)

# Save the processed datasets
train_df.to_csv(os.path.join(data_dir_1, 'train.csv'), index=False)
val_df.to_csv(os.path.join(data_dir_1, 'val.csv'), index=False)
test_df.to_csv(os.path.join(data_dir_1, 'test.csv'), index=False)
train_df_dropped.to_csv(os.path.join(data_dir_2, 'train.csv'), index=False)
val_df_dropped.to_csv(os.path.join(data_dir_2, 'val.csv'), index=False)
test_df_dropped.to_csv(os.path.join(data_dir_2, 'test.csv'), index=False)

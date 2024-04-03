import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, get_feature_names
from tensorflow.keras.optimizers import Adam

from scripts.BERT.Eval import calculate_ndcg

# Load Data
def load_data():
    train = pd.read_csv('data/DeepFM/Concurrent/train.csv')
    val = pd.read_csv('data/DeepFM/Concurrent/val.csv')
    test = pd.read_csv('data/DeepFM/Concurrent/test.csv')
    
    train['set'] = 'train'
    val['set'] = 'val'
    test['set'] = 'test'
    
    return pd.concat([train, val, test])

# Preprocess Data
def preprocess_data(data):
    sparse_features = ['C' + str(i) for i in range(1, 11)]
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    return data, sparse_features

# Prepare Model Input
def prepare_model_input(data, sparse_features):
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4) for feat in sparse_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    model_input = {name: data[name].values for name in feature_names}
    return model_input, feature_names


def split_data(combined_data):
    # Split combined dataset back into train, validation, and test sets
    train_data = combined_data[combined_data['set'] == 'train'].drop('set', axis=1)
    val_data = combined_data[combined_data['set'] == 'val'].drop('set', axis=1)
    test_data = combined_data[combined_data['set'] == 'test'].drop('set', axis=1)
    return train_data, val_data, test_data


# Train Model
def train_model(train_input, val_input, feature_names):
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['binary_crossentropy'])
    
    history = model.fit(train_input, train_input['label'].values,
                        batch_size=256, epochs=10, verbose=2,
                        validation_data=(val_input, val_input['label'].values))
    return model

# Evaluate Model
def evaluate_model(model, test_input, feature_names):
    test_input['score'] = model.predict(test_input, batch_size=256)
    test_input['group'] = np.arange(len(test_input)) // 50
    grouped_test = test_input.groupby('group')

    ndcg_scores = grouped_test.apply(lambda x: calculate_ndcg(x, k=10))
    print(f"Mean NDCG@10: {ndcg_scores.mean()}")

    ndcg_scores = grouped_test.apply(lambda x: calculate_ndcg(x, k=5))
    print(f"Mean NDCG@5: {ndcg_scores.mean()}")

    test_input['binary_predictions'] = test_input['score'].apply(lambda x: 1 if x >= 0.5 else 0)
    accuracy = accuracy_score(test_input['label'].values, test_input['binary_predictions'].values)
    f1 = f1_score(test_input['label'].values, test_input['binary_predictions'].values)

    print(f"Accuracy: {accuracy}\nF1 Score: {f1}")

# Main
combined_data = load_data()
combined_data, sparse_features = preprocess_data(combined_data)

train_data, val_data, test_data = split_data(combined_data)  # Split the combined dataset

train_model_input, feature_names = prepare_model_input(train_data, sparse_features)
val_model_input, _ = prepare_model_input(val_data, sparse_features)
test_model_input, _ = prepare_model_input(test_data, sparse_features)

model = train_model(train_model_input, val_model_input, feature_names)
evaluate_model(model, test_model_input, feature_names)

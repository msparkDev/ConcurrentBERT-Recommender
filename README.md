# Enhancing E-commerce Recommendation Systems with Concurrent Purchase Data: A Transformer-Based Approach

## Authors
- Minseo Park
- Jangmin Oh*

This project introduces a novel approach to enhance e-commerce recommendation systems by incorporating concurrent purchase data alongside transformer-based models like BERT. Our methodology demonstrates significant improvements in predictive accuracy and overall system efficacy, as evidenced by extensive validation on real-world data from Katcher's e-commerce platform.

## Key Contributions
1. **Integration of Concurrent Purchase Data**: Our approach uniquely integrates concurrent purchase data, offering new insights into consumer behavior patterns.
2. **Transformer-Based Algorithm**: We employ a transformer-based recommendation algorithm, significantly outperforming traditional models in predicting next-product purchases.
3. **Real-World Validation**: Rigorous testing on Katchers' data validates our method's effectiveness in enhancing recommendation accuracy.

## Dependencies
To replicate our work or utilize our framework, ensure the installation of the following:
- Python 3.10 or higher
- Required libraries and packages, installable via `pip install -r requirements.txt` in your terminal.

## Data Privacy and Usage
Due to proprietary restrictions, the dataset from Katchers' e-commerce platform is not publicly available. However, to facilitate understanding and reproducibility, we utilize the [**Online Retail Dataset**](https://archive.ics.uci.edu/ml/datasets/Online+Retail) from the UCI Machine Learning Repository as a demonstrative proxy.

### Alternative Dataset for Demonstration
- **Online Retail Dataset**: Contains comprehensive transaction records, making it an ideal candidate for research and development in e-commerce recommendation systems.

## Implementation
Our project repository is structured to guide you through setting up the environment, preprocessing the data, and executing the recommendation models.

### Setup and Installation
Clone the repository and install dependencies as follows:
```bash
git clone https://github.com/msparkDev/ECommTransformerRecSys.git
cd ECommTransformerRecSys
pip install -r requirements.txt
```

## Data Preparation
Our framework considers both scenarios: with and without concurrent purchases, for data preprocessing. To facilitate ease of use and immediate experimentation, we have preprocessed the [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail) from the UCI Machine Learning Repository following both scenarios. The preprocessed data is readily available in the repository, allowing you to skip manual preprocessing steps and dive directly into exploring the recommendation models.

### Preprocessed Data Availability
The preprocessed datasets are stored within the repository, under the following structure:
- For data considering concurrent purchases: `data/BERT_ConcurrentPurchases`
- For data not considering concurrent purchases: `data/BERT_SinglePurchases`

Within these directories, you will find:
- `train_data.csv`, `validation_data.csv`, `test_data.csv`: Splits of the raw data into training, validation, and testing sets.
- `negative_train.csv`, `negative_val.csv`, `negative_test.csv`: Negative samples for each set, generated through negative sampling.
- Final datasets for BERT inputs:
  - `trainForBERT_WCP.csv`, `valForBERT_WCP.csv`, `testForBERT_WCP.csv` for scenarios with concurrent purchases.
  - `trainForBERT_WOCP.csv`, `valForBERT_WOCP.csv`, `testForBERT_WOCP.csv` for scenarios without concurrent purchases.

### Generating Data
If you wish to regenerate the data or apply preprocessing to a new dataset, execute the following scripts:

##### Generates data with concurrent purchase considerations and stores it in data/BERT_ConcurrentPurchases
```bash
python scripts/BERT-RecSysWithConcurrentDataPreparation.py
```

##### Generates data without concurrent purchase considerations and stores it in data/BERT_SinglePurchases
```bash
python scripts/BERT-RecSysWithoutConcurrentDataPreparation.py
```



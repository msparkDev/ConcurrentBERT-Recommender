# Enhancing E-commerce Recommendation Systems with Concurrent Purchase Data: A Transformer-Based Approach

## Authors
- Minseo Park
- Jangmin Oh*

## Abstract
This project introduces a novel approach to enhance e-commerce recommendation systems by incorporating concurrent purchase data alongside transformer-based models like BERT. Our methodology demonstrates significant improvements in predictive accuracy and overall system efficacy, as evidenced by extensive validation on real-world data from Katcher's e-commerce platform.

## Introduction
E-commerce platforms continually seek advanced recommendation systems to enhance user experience and business outcomes. Traditional systems often overlook the rich contextual information available from concurrent purchases. We propose a method that integrates this data, leveraging the BERT model to understand and predict complex consumer purchasing behaviors more accurately.

## Key Contributions
1. **Integration of Concurrent Purchase Data**: Our approach uniquely integrates concurrent purchase data, offering new insights into consumer behavior patterns.
2. **Transformer-Based Algorithm**: We employ a transformer-based recommendation algorithm, significantly outperforming traditional models in predicting next-product purchases.
3. **Real-World Validation**: Rigorous testing on Katcher's data validates our method's effectiveness in enhancing recommendation accuracy.

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

## Data Preparation
To preprocess the data, considering both scenarios with and without concurrent purchases, execute the following scripts:

```bash
python scripts/BERT-RecSysWithConcurrentDataPreparation.py
python scripts/BERT-RecSysWithoutConcurrentDataPreparation.py
```

# Enhancing e-commerce recommendation systems with concurrent purchase data: a BERT-based approach
Minseo Park, Jangmin Oh*
<img src="https://github.com/msparkDev/ECommTransformerRecSys/blob/main/image.png" width="3000">
A workflow that recommends products that the user is likely to purchase next. Because our method handles the user's concurrent purchases, it better analyzes the user's more complex patterns and shows higher performance than other methods.

### Our novel contributions are:
1. Developed a unique system to **integrate concurrent purchase data into e-commerce recommendation models,** enhancing understanding of complex buying patterns.
2. Applied the BERT transformer model to analyze e-commerce order histories as natural language, capturing deeper product relationships.
3. Demonstrated through real-world data that **our approach significantly outperforms traditional recommendation systems** in accuracy and predictive performance.

## Dependencies
To get started with the framework, install the following dependencies:
- [Python 3.10](https://www.python.org/)
- `pip install -r requirments.txt`

## Data
Due to the proprietary nature of Katchers' dataset, we are unable to make the entire dataset publicly available. However, for demonstration purposes, we have sampled 100 entries from our training data, which can be found at **data/samples/katchers_data.csv** in our repository.

Instead, UCI Online Retail Dataset is publicly available as open-source data, allowing us to share both our data generation process and the outcomes. Follow the commands below to build the data and the resulting processed data are stored in the **data folder** of our GitHub repository.

### BERT with Concurrent Purchases
```
python scripts/BERT/DataPrepWith.py
```

### Optional Scripts
- BERT without Concurrent Purchases
  
  ```
  python scripts/BERT/DataPrepWithout.py
  ```

- DeepFM & XGBoost (with/without Concurrent Purchases)
  
  ```
  python scripts/DeepFM/DataPrep.py
  ```
  
## Training & Inference
### BERT with Concurrent Purchases
0. HuggingFace Login
```
export HUGGINGFACE_CO='your_huggingface_token_here'
```
1. Training
```
python scripts/BERT/Train.py
```
- Make sure to replace **YourUsernameHere** with your actual HuggingFace account username in the **output_dir** parameter.
2. Inference
```
python scripts/BERT/Eval.py
```

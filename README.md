# Enhancing E-commerce Recommendation Systems with Concurrent Purchase Data: A Transformer-Based Approach

## Authors
Minseo Park, Jangmin Oh*

This project presents a novel approach to improving e-commerce recommendation systems by leveraging concurrent purchase data and transformer-based models. Our work demonstrates significant advancements in predictive accuracy and system efficacy, validated with real-world data.

## Key Contributions

1. **Integration of Concurrent Purchase Data:** We introduce an innovative method by incorporating concurrent purchase data into e-commerce recommendation systems. This approach allows for a deeper understanding of complex consumer purchasing patterns, significantly enhancing the accuracy of predictions.

2. **Transformer-Based Recommendation Algorithm:** Utilizing the BERT model, we have developed a transformer-based algorithm specifically fine-tuned for predicting the next product a customer is likely to purchase. This method represents a considerable leap forward compared to traditional recommendation systems.

3. **Validation on Real-World Data:** Our approach has been rigorously tested using data from Katcher's e-commerce platform. The results showcase our method's superior performance in making accurate predictions, thereby establishing a new standard for recommendation system efficacy.

## Getting Started

### Dependencies

Ensure you have the following installed to use this framework:

- Python 3.10
- Required Python packages (install using the command below)

```bash
pip install -r requirements.txt
```

## Data

### Note on Data Privacy
The dataset used in this project is proprietary to Katcher's and, as such, cannot be made publicly available. We prioritize the privacy and confidentiality of this data.

### Alternative Dataset for Demonstration

For demonstration purposes, we use the [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail) from the UCI Machine Learning Repository. This open-source dataset helps illustrate our methodologies and the application of our framework:

- **Online Retail Dataset:** This dataset contains transaction records from a UK-based online retail platform, covering a period from December 1st, 2010, to December 9th, 2011. It includes various features such as `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`, making it suitable for e-commerce recommendation system research and development.

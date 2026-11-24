
# E-Commerce Customer Analytics

## Project Overview

This project analyses e-commerce customer purchasing
behaviour using Python, Data Science and Machine Learning.

An interactive Streamlit dashboard displays business
insights, customer segmentation, spending predictions,
purchase probabilities and product recommendations.

## Project Objectives

- Clean and preprocess real-world customer data.
- Analyse customer purchasing behaviour.
- Visualize sales trends and popular products.
- Segment customers into different groups.
- Predict future customer spending.
- Predict repeat purchase probability.
- Recommend similar products.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- VS Code

## Dataset

UCI Online Retail Dataset

Source:
https://archive.ics.uci.edu/dataset/352/online-retail

License: CC BY 4.0

Dataset DOI: 10.24432/C5BW33

- Original records: 541,909
- Cleaned records: 392,692
- Original columns: 8
- Processed columns: 10

## Machine Learning Algorithms

| Algorithm | Purpose |
|---|---|
| K-Means Clustering | Customer segmentation |
| Linear Regression | Customer spending prediction |
| Logistic Regression | Repeat purchase prediction |
| KNN | Product recommendation |

## Project Results

- Successfully cleaned 392,692 transaction records.
- Created Premium, Regular and Occasional segments.
- Achieved a K-Means Silhouette Score of 0.337.
- Generated customer spending predictions.
- Generated customer purchase probabilities.
- Generated 2,500 recommendations for 500 products.
- Developed an interactive Streamlit dashboard.

## Installation

Install the required Python libraries:

    pip install -r requirements.txt

Download the UCI Online Retail dataset and place
Online Retail.xlsx inside the main project folder.

Run the Python files in the following order:

    python main.py
    python data_cleaning.py
    python data_analysis.py
    python customer_segmentation.py
    python spending_prediction.py
    python purchase_prediction.py
    python product_recommendation.py

## Run the Dashboard

    python -m streamlit run app.py

Open the browser:

http://localhost:8501

## Limitations

- The dataset contains historical transactions.
- Browsing history and customer ratings are unavailable.
- Predictions are based on a historical test period.
- Recommendations are based on purchase similarity.
- Further model optimization can improve performance.

## Future Scope

- Real-time customer analytics.
- Advanced machine learning models.
- Personalized product recommendations.
- Integration with live e-commerce platforms.
- Cloud deployment.

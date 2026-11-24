
import streamlit as st
import pandas as pd

# Website configuration
st.set_page_config(
    page_title="E-Commerce Customer Analytics",
    page_icon="🛒",
    layout="wide"
)

# Load datasets
@st.cache_data
def load_data():
    return (
        pd.read_csv("cleaned_data.csv"),
        pd.read_csv("customer_segments.csv"),
        pd.read_csv("spending_predictions.csv"),
        pd.read_csv("purchase_predictions.csv"),
        pd.read_csv(
            "product_recommendations.csv",
            dtype={"ProductCode": str,
                   "RecommendedCode": str}
        )
    )

df, segments, spending, purchase, recommendations = load_data()

# Sidebar navigation
st.sidebar.title("🛒 E-Commerce Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Sales Analysis",
        "Customer Segmentation",
        "Spending Prediction",
        "Purchase Prediction",
        "Product Recommendation"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Data Science and Machine Learning Project"
)


# PAGE 1: HOME

if page == "Home":

    st.title("🛒 E-Commerce Customer Analytics")

    st.subheader(
        "Machine Learning and Business Intelligence"
    )

    st.write(
        "This project analyses customer purchasing "
        "behaviour using data science and machine learning."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Sales",
        f"£{df['TotalAmount'].sum():,.2f}"
    )

    col2.metric(
        "Total Customers",
        df["CustomerID"].nunique()
    )

    col3.metric(
        "Total Orders",
        df["InvoiceNo"].nunique()
    )

    col4.metric(
        "Total Products",
        df["StockCode"].nunique()
    )

    st.divider()

    st.subheader("Project Objectives")

    st.markdown("""
    - Analyse customer purchasing behaviour
    - Discover sales patterns and trends
    - Segment customers using K-Means
    - Predict spending using Linear Regression
    - Predict purchases using Logistic Regression
    - Recommend products using KNN
    """)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.caption(
        "Dataset: UCI Online Retail Dataset"
    )


# PAGE 2: SALES ANALYSIS

elif page == "Sales Analysis":

    st.title("📊 Sales Analysis")

    st.subheader("Monthly Sales Trend")
    st.image("graphs/monthly_sales.png")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Selling Products")
        st.image("graphs/top_products.png")

    with col2:
        st.subheader("Country-wise Sales")
        st.image("graphs/country_sales.png")

    st.subheader("Top Customers by Spending")
    st.image("graphs/top_customers.png")


# PAGE 3: CUSTOMER SEGMENTATION

elif page == "Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.write("Algorithm: K-Means Clustering")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Premium Customers",
        int((segments["Segment"] == "Premium").sum())
    )

    col2.metric(
        "Regular Customers",
        int((segments["Segment"] == "Regular").sum())
    )

    col3.metric(
        "Occasional Customers",
        int((segments["Segment"] == "Occasional").sum())
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image("graphs/customer_segmentation.png")

    with col2:
        st.image("graphs/segment_distribution.png")

    st.subheader("Customer Segment Summary")

    summary = segments.groupby("Segment")[
        ["Recency", "Frequency", "Monetary"]
    ].mean().round(2)

    st.dataframe(summary)

    selected = st.selectbox(
        "Select Customer Segment",
        ["All", "Premium", "Regular", "Occasional"]
    )

    if selected == "All":
        filtered = segments
    else:
        filtered = segments[
            segments["Segment"] == selected
        ]

    st.dataframe(
        filtered,
        use_container_width=True
    )


# PAGE 4: SPENDING PREDICTION

elif page == "Spending Prediction":

    st.title("💰 Customer Spending Prediction")

    st.write("Algorithm: Linear Regression")

    st.info(
        "Predicted customer spending over a 30-day "
        "historical test period."
    )

    customer_id = st.selectbox(
        "Select Customer ID",
        sorted(spending["CustomerID"].unique())
    )

    customer = spending[
        spending["CustomerID"] == customer_id
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Spending",
        f"£{customer['PredictedSpending']:,.2f}"
    )

    col2.metric(
        "Actual Spending",
        f"£{customer['ActualSpending']:,.2f}"
    )

    col3.metric(
        "Prediction Error",
        f"£{abs(customer['ActualSpending'] - customer['PredictedSpending']):,.2f}"
    )

    st.subheader("Customer Purchase History")

    st.write("Recency:", int(customer["Recency"]), "days")
    st.write("Frequency:", int(customer["Frequency"]), "orders")
    st.write("Historical Spending:", 
             f"£{customer['Monetary']:,.2f}")

    st.image("graphs/spending_prediction.png")

    st.subheader("All Spending Predictions")

    st.dataframe(
        spending,
        use_container_width=True
    )


# PAGE 5: PURCHASE PREDICTION

elif page == "Purchase Prediction":

    st.title("🎯 Customer Purchase Prediction")

    st.write("Algorithm: Logistic Regression")

    customer_id = st.selectbox(
        "Select Customer ID",
        sorted(purchase["CustomerID"].unique())
    )

    customer = purchase[
        purchase["CustomerID"] == customer_id
    ].iloc[0]

    probability = float(
        customer["PurchaseProbability"]
    )

    st.metric(
        "Purchase Probability",
        f"{probability:.2f}%"
    )

    st.progress(
        max(0.0, min(probability / 100, 1.0))
    )

    if customer["PredictedPurchase"] == 1:
        st.success("Prediction: Likely to Purchase")
    else:
        st.warning("Prediction: Unlikely to Purchase")

    actual = (
        "Purchased"
        if customer["ActualPurchase"] == 1
        else "Did Not Purchase"
    )

    st.write("Actual Result:", actual)

    st.subheader("Confusion Matrix")

    st.image("graphs/confusion_matrix.png")

    st.subheader("All Purchase Predictions")

    st.dataframe(
        purchase,
        use_container_width=True
    )


# PAGE 6: PRODUCT RECOMMENDATION

elif page == "Product Recommendation":

    st.title("🛍️ Product Recommendation System")

    st.write("Algorithm: K-Nearest Neighbors (KNN)")

    st.write(
        "Select a product to discover five products "
        "with similar customer purchasing patterns."
    )

    products = recommendations[
        ["ProductCode", "Product"]
    ].drop_duplicates("ProductCode")

    product_names = dict(
        zip(products["ProductCode"], products["Product"])
    )

    selected_code = st.selectbox(
        "Select a Product",
        products["ProductCode"].tolist(),
        format_func=lambda code: (
            f"{product_names[code]} ({code})"
        )
    )

    results = recommendations[
        recommendations["ProductCode"] == selected_code
    ].sort_values("Similarity", ascending=False)

    st.subheader("Recommended Products")

    for rank, (_, row) in enumerate(
        results.iterrows(), start=1
    ):

        st.markdown(
            f"**{rank}. {row['RecommendedProduct']}**"
        )

        st.write(
            f"Similarity Score: {row['Similarity']:.2f}%"
        )

        st.progress(
            max(0.0, min(float(row["Similarity"]) / 100, 1.0))
        )

        st.divider()

    st.caption(
        "Similarity is based on historical purchasing "
        "patterns, not purchase probability."
    )


# Footer

st.divider()

st.caption(
    "E-Commerce Customer Analytics | "
    "Python | Machine Learning | Streamlit"
)

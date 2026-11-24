
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Separate past and future transactions
end_date = df["InvoiceDate"].max().normalize()
end_date = end_date + pd.Timedelta(days=1)

cutoff_date = end_date - pd.Timedelta(days=30)

past_data = df[df["InvoiceDate"] < cutoff_date]
future_data = df[df["InvoiceDate"] >= cutoff_date]

# Create historical customer features
customers = past_data.groupby("CustomerID").agg(
    Recency=("InvoiceDate",
             lambda x: (cutoff_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalAmount", "sum"),
    TotalQuantity=("Quantity", "sum")
)

# Calculate actual spending during the next 30 days
future_spending = future_data.groupby(
    "CustomerID"
)["TotalAmount"].sum()

customers["FutureSpending"] = future_spending.reindex(
    customers.index, fill_value=0
)

# Input features
X = customers[
    ["Recency", "Frequency", "Monetary", "TotalQuantity"]
]

# Target variable
y = customers["FutureSpending"]

# Split data: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict customer spending
predictions = model.predict(X_test)

# Prevent negative spending predictions
predictions = np.maximum(predictions, 0)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- LINEAR REGRESSION RESULTS ---")

print("Training Customers:", len(X_train))
print("Testing Customers:", len(X_test))

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Compare with a simple baseline
baseline = np.full(len(y_test), y_train.mean())

print("Baseline MAE:", round(
    mean_absolute_error(y_test, baseline), 2
))

# Save prediction results
results = X_test.copy()

results["ActualSpending"] = y_test
results["PredictedSpending"] = predictions.round(2)

results.to_csv("spending_predictions.csv")

# Save customer features for Logistic Regression
customers.to_csv("customer_features.csv")

print("\nSample Predictions:")
print(results[
    ["ActualSpending", "PredictedSpending"]
].head(10))

# Actual vs predicted spending graph
plt.figure(figsize=(10, 6))

plt.scatter(
    y_test, predictions,
    color="blue", alpha=0.5
)

limit = max(y_test.max(), predictions.max())

plt.plot(
    [0, limit], [0, limit],
    color="red", linestyle="--"
)

plt.title("Actual vs Predicted Customer Spending")
plt.xlabel("Actual Spending (£)")
plt.ylabel("Predicted Spending (£)")
plt.tight_layout()

plt.savefig("graphs/spending_prediction.png")
plt.close()

print("\nSpending prediction completed successfully!")

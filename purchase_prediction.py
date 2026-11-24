
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# Load customer features
df = pd.read_csv("customer_features.csv")

# Create target column
# 1 = Purchased, 0 = Did not purchase
df["WillPurchase"] = (
    df["FutureSpending"] > 0
).astype(int)

# Input features
X = df[
    ["Recency", "Frequency", "Monetary", "TotalQuantity"]
]

# Target variable
y = df["WillPurchase"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Logistic Regression model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# Train the model
model.fit(X_train, y_train)

# Predict purchase decisions
predictions = model.predict(X_test)

# Predict purchase probability
probabilities = model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

# 5-fold cross-validation
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\n--- LOGISTIC REGRESSION RESULTS ---")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("ROC AUC Score:", round(auc, 4))

print("Cross Validation Accuracy:",
      round(cv_scores.mean() * 100, 2), "%")

print("Majority Class Baseline:",
      round(y_test.value_counts(normalize=True).max() * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save prediction results
results = X_test.copy()

results["CustomerID"] = df.loc[
    X_test.index, "CustomerID"
]

results["ActualPurchase"] = y_test
results["PredictedPurchase"] = predictions

results["PurchaseProbability"] = (
    probabilities * 100
).round(2)

results.to_csv(
    "purchase_predictions.csv",
    index=False
)

print("\nSample Purchase Predictions:")

print(results[
    ["CustomerID", "ActualPurchase",
     "PredictedPurchase", "PurchaseProbability"]
].head(10))

# Generate confusion matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Purchase", "Purchase"],
    yticklabels=["No Purchase", "Purchase"]
)

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Result")
plt.ylabel("Actual Result")
plt.tight_layout()

plt.savefig("graphs/confusion_matrix.png")
plt.close()

print("\nPurchase prediction completed successfully!")

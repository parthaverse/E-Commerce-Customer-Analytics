
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Convert invoice date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Set reference date
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Calculate RFM values for each customer
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", 
             lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalAmount", "sum")
)

print("\nRFM Analysis:")
print(rfm.head())

# Reduce the effect of extreme values
features = np.log1p(rfm)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Apply K-Means clustering
rfm["Cluster"] = kmeans.fit_predict(scaled_data)

# Rank clusters according to median spending
ranking = rfm.groupby("Cluster")["Monetary"].median().sort_values()

segment_names = {
    ranking.index[0]: "Occasional",
    ranking.index[1]: "Regular",
    ranking.index[2]: "Premium"
}

# Assign customer segment names
rfm["Segment"] = rfm["Cluster"].map(segment_names)

# Display customer segment counts
print("\nCustomer Segmentation:")
print(rfm["Segment"].value_counts())

# Display average customer behaviour
print("\nSegment Summary:")
print(rfm.groupby("Segment")[
    ["Recency", "Frequency", "Monetary"]
].mean().round(2))

# Evaluate the model
score = silhouette_score(scaled_data, rfm["Cluster"])

print("\nSilhouette Score:", round(score, 3))

# Save customer segmentation results
rfm.to_csv("customer_segments.csv")

# Graph 1: Customer segmentation
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="Segment",
    palette="Set1",
    alpha=0.7
)

plt.xscale("log")
plt.yscale("log")
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Purchase Frequency")
plt.ylabel("Total Spending (£)")
plt.tight_layout()
plt.savefig("graphs/customer_segmentation.png")
plt.close()

# Graph 2: Customer segment distribution
plt.figure(figsize=(8, 8))

rfm["Segment"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=["skyblue", "orange", "lightgreen"]
)

plt.title("Customer Segment Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("graphs/segment_distribution.png")
plt.close()

print("\nCustomer segmentation completed successfully!")

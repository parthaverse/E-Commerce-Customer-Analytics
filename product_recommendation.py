
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load cleaned dataset
df = pd.read_csv(
    "cleaned_data.csv",
    dtype={"StockCode": str}
)

# Select 500 popular products
popular_products = df["StockCode"].value_counts().head(500).index

data = df[df["StockCode"].isin(popular_products)]

# Create product-customer matrix
product_matrix = data.pivot_table(
    index="StockCode",
    columns="CustomerID",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

# Convert quantities into purchase indicators
# 1 = Purchased, 0 = Not purchased
product_matrix = (product_matrix > 0).astype(int)

print("Product Matrix Shape:", product_matrix.shape)

# Create KNN model
model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=6
)

# Train the model
model.fit(product_matrix)

# Find similar products
distances, indices = model.kneighbors(product_matrix)

# Map product codes to product names
product_names = df.drop_duplicates(
    "StockCode"
).set_index("StockCode")["Description"]

recommendations = []

# Generate recommendations for every product
for i, product_code in enumerate(product_matrix.index):

    count = 0

    for distance, index in zip(distances[i], indices[i]):

        recommended_code = product_matrix.index[index]

        # Skip the selected product itself
        if recommended_code == product_code:
            continue

        similarity = (1 - distance) * 100

        recommendations.append({
            "ProductCode": product_code,
            "Product": product_names[product_code],
            "RecommendedCode": recommended_code,
            "RecommendedProduct": product_names[recommended_code],
            "Similarity": round(similarity, 2)
        })

        count += 1

        if count == 5:
            break

# Convert recommendations into DataFrame
results = pd.DataFrame(recommendations)

# Save recommendations
results.to_csv(
    "product_recommendations.csv",
    index=False
)

# Display sample recommendations
sample_code = product_matrix.index[0]

sample = results[
    results["ProductCode"] == sample_code
]

print("\n--- KNN PRODUCT RECOMMENDATION ---")

print("\nSelected Product:")
print(product_names[sample_code])

print("\nRecommended Products:")

print(sample[
    ["RecommendedProduct", "Similarity"]
].to_string(index=False))

print("\nTotal Products:", len(product_matrix))

print("Total Recommendations:", len(results))

print("\nProduct recommendation completed successfully!")

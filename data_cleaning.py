
import pandas as pd

# Load the original dataset
df = pd.read_excel("Online Retail.xlsx")

print("Original dataset shape:", df.shape)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove missing customer IDs and product descriptions
df = df.dropna(subset=["CustomerID", "Description"])

# Remove duplicate records
df = df.drop_duplicates()

# Remove cancelled orders
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Remove invalid quantities and prices
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

# Convert customer ID into integer
df["CustomerID"] = df["CustomerID"].astype(int)

# Convert invoice date into datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Calculate total purchase amount
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# Create month column
df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

# Save the cleaned dataset
df.to_csv("cleaned_data.csv", index=False)

print("\nCleaned dataset shape:", df.shape)
print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nData cleaning completed successfully!")

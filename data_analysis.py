
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Automatically create a folder for graphs
os.makedirs("graphs", exist_ok=True)

sns.set_theme(style="whitegrid")

# Basic business analysis
print("\n--- BUSINESS SUMMARY ---")

print("Total Sales: £", round(df["TotalAmount"].sum(), 2))
print("Total Customers:", df["CustomerID"].nunique())
print("Total Orders:", df["InvoiceNo"].nunique())
print("Total Products:", df["StockCode"].nunique())
print("Total Countries:", df["Country"].nunique())


# Graph 1: Monthly Sales Trend

monthly_sales = df.groupby("Month")["TotalAmount"].sum()

plt.figure(figsize=(12, 5))
monthly_sales.plot(kind="line", marker="o", color="blue")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales (£)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graphs/monthly_sales.png")
plt.close()


# Graph 2: Top 10 Best-Selling Products

top_products = df.groupby("Description")["Quantity"].sum()
top_products = top_products.nlargest(10)

plt.figure(figsize=(12, 6))
top_products.sort_values().plot(kind="barh", color="green")
plt.title("Top 10 Best-Selling Products")
plt.xlabel("Quantity Sold")
plt.tight_layout()
plt.savefig("graphs/top_products.png")
plt.close()


# Graph 3: Top 10 Countries by Sales

country_sales = df.groupby("Country")["TotalAmount"].sum()
country_sales = country_sales.nlargest(10)

plt.figure(figsize=(10, 6))
country_sales.sort_values().plot(kind="barh", color="orange")
plt.title("Top 10 Countries by Sales")
plt.xlabel("Total Sales (£)")
plt.tight_layout()
plt.savefig("graphs/country_sales.png")
plt.close()


# Graph 4: Top 10 Customers by Spending

top_customers = df.groupby("CustomerID")["TotalAmount"].sum()
top_customers = top_customers.nlargest(10)

plt.figure(figsize=(10, 6))
top_customers.plot(kind="bar", color="purple")
plt.title("Top 10 Customers by Spending")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending (£)")
plt.tight_layout()
plt.savefig("graphs/top_customers.png")
plt.close()


print("\nData analysis completed successfully!")
print("All 4 graphs saved inside the graphs folder.")

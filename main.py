import pandas as pd

# Load the dataset
df = pd.read_excel("Online Retail.xlsx")

# Display first 5 records
print(df.head())

# Display number of rows and columns
print("Dataset Shape:", df.shape)

# Display column names
print("Columns:", df.columns.tolist())

import pandas as pd

# Step 1: Load the three raw datasets

customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")

# Step 2: Create a dictionary to manage all datasets

datasets = {
    "Customers": customers,
    "Products": products,
    "Orders": orders
}

# Step 3: Audit each dataset

for name, df in datasets.items():

    print("\n================================")
    print("DATASET:", name)
    print("================================")

    # Number of rows and columns
    print("\nShape:")
    print(df.shape)

    # Column names and data types
    print("\nData Types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Missing value percentages
    print("\nMissing Value Percentage:")
    print((df.isnull().sum() / len(df)) * 100)

    print("\nFirst 5 Rows:")
    print(df.head())

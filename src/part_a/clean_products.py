
import pandas as pd

# Load original products
products = pd.read_csv("data/raw/products.csv")

# Basic validation
print("Original shape:", products.shape)
print("Missing values:")
print(products.isnull().sum())

# Save the unchanged product data
products.to_csv("data/clean/products.csv", index=False)

print("\nClean products saved successfully!")
print("Final shape:", products.shape)

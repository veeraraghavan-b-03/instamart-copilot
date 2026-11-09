
import sqlite3
import pandas as pd

# 1. Load the three cleaned datasets
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

print("Loaded cleaned datasets:")
print("Customers:", customers.shape)
print("Products:", products.shape)
print("Orders:", orders.shape)


# 2. Convert order_ts to the required SQLite text format
orders["order_ts"] = (
    pd.to_datetime(orders["order_ts"])
    .dt.strftime("%Y-%m-%d %H:%M:%S")
)


# 3. Create SQLite database and store the datasets
database_path = "data/instamart.db"

with sqlite3.connect(database_path) as connection:

    customers.to_sql(
        "customers",
        connection,
        if_exists="replace",
        index=False
    )

    products.to_sql(
        "products",
        connection,
        if_exists="replace",
        index=False
    )

    orders.to_sql(
        "orders",
        connection,
        if_exists="replace",
        index=False
    )

    # 4. Verify the number of records in each table
    print("\n========== DATABASE VERIFICATION ==========")

    for table in ["customers", "products", "orders"]:
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS total FROM {table}",
            connection
        )

        print(f"{table}: {count.iloc[0]['total']} records")

print("\nSQLite database created successfully!")
print("Database location:", database_path)

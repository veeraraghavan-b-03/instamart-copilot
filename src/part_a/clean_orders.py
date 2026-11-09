

import pandas as pd

# --------------------------------------------------
# 1. Load the raw orders and products datasets
# --------------------------------------------------

orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/raw/products.csv")

print("Original orders shape:", orders.shape)


# --------------------------------------------------
# 2. Convert order_ts into datetime format
# --------------------------------------------------

orders["order_ts"] = pd.to_datetime(orders["order_ts"])


# --------------------------------------------------
# 3. Remove exact duplicate rows
# --------------------------------------------------

orders = orders.drop_duplicates()

print("After removing exact duplicates:", orders.shape)


# --------------------------------------------------
# 4. Keep the latest record for each order_id
# --------------------------------------------------

orders = orders.sort_values("order_ts")

orders = orders.drop_duplicates(
    subset="order_id",
    keep="last"
)

print("After removing repeated order IDs:", orders.shape)


# --------------------------------------------------
# 5. Convert order_amount into numeric format
# --------------------------------------------------

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)


# --------------------------------------------------
# 6. Get product price and category
# --------------------------------------------------

product_lookup = products[
    ["product_id", "price_inr", "category"]
]

orders = orders.merge(
    product_lookup,
    on="product_id",
    how="left",
    validate="many_to_one"
)


# --------------------------------------------------
# 7. Fill missing order amounts using product price
# --------------------------------------------------

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount.round(2)
)


# --------------------------------------------------
# 8. Set negative delivery times to missing
# --------------------------------------------------

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA


# --------------------------------------------------
# 9. Fill missing delivery times for completed orders
#    using the median of their product category
# --------------------------------------------------

completed_mask = orders["status"].str.lower().eq("completed")

category_median = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"]
    .fillna(category_median)
)


# --------------------------------------------------
# 10. Remove temporary product lookup columns
# --------------------------------------------------

orders = orders.drop(columns=["price_inr", "category"])


# --------------------------------------------------
# 11. Save the cleaned orders dataset
# --------------------------------------------------

orders.to_csv("data/clean/orders.csv", index=False)


# --------------------------------------------------
# 12. Display cleaning results
# --------------------------------------------------

print("\n========== ORDER CLEANING RESULTS ==========")

print("Final shape:", orders.shape)

print("\nOrder status counts:")
print(orders["status"].value_counts())

print("\nMissing order amounts:", orders["order_amount"].isna().sum())

print(
    "Missing delivery times in completed orders:",
    orders.loc[completed_mask, "delivery_minutes"].isna().sum()
)

print(
    "Missing delivery times in cancelled orders:",
    orders.loc[orders["status"].str.lower().eq("cancelled"),
               "delivery_minutes"].isna().sum()
)

print(
    "\nCompleted order revenue:",
    round(
        orders.loc[completed_mask, "order_amount"].sum(),
        2
    )
)

print("\nCleaned orders dataset saved successfully!")

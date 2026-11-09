import pandas as pd

# Load cleaned data
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Keep only completed orders
completed_orders = orders[
    orders["status"] == "completed"
]

# Join orders with customers and products
analysis = completed_orders.merge(
    customers,
    on="customer_id"
).merge(
    products,
    on="product_id"
)

# Revenue by city and category
pivot = pd.pivot_table(
    analysis,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Grand Total"
)

print("========== PANDAS PIVOT TABLE ==========")
print(pivot.round(2))

print("\nGrand Total Revenue:")
print(round(analysis["order_amount"].sum(), 2))


# City-wise revenue statistics
city_stats = completed_orders.merge(
    customers[["customer_id", "city"]],
    on="customer_id"
)

city_summary = city_stats.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    average_order_value=("order_amount", "mean"),
    completed_orders=("order_id", "count")
)

print("\n========== CITY-WISE REVENUE STATISTICS ==========")
print(city_summary.round(2))


# Distribution and outlier analysis
amounts = completed_orders["order_amount"]

# Basic statistics
mean_value = amounts.mean()
median_value = amounts.median()
skew_value = amounts.skew()

# Quartiles
q1 = amounts.quantile(0.25)
q3 = amounts.quantile(0.75)

# Interquartile range
iqr = q3 - q1

# Outlier boundaries
lower_fence = q1 - (1.5 * iqr)
upper_fence = q3 + (1.5 * iqr)

# Count outliers
lower_outliers = (amounts < lower_fence).sum()
upper_outliers = (amounts > upper_fence).sum()

print("\n========== ORDER AMOUNT DISTRIBUTION ==========")
print("Mean:", round(mean_value, 2))
print("Median:", round(median_value, 2))
print("Skewness:", round(skew_value, 2))
print("Q1:", round(q1, 2))
print("Q3:", round(q3, 2))
print("IQR:", round(iqr, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Lower outliers:", lower_outliers)
print("Upper outliers:", upper_outliers)

# Correlation analysis
correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation_data = completed_orders[correlation_columns]

correlation_matrix = correlation_data.corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation_matrix.round(3))

# Find the strongest correlation pair
correlation_pairs = (
    correlation_matrix
    .where(
        ~pd.DataFrame(
            [[i == j for j in range(len(correlation_columns))]
             for i in range(len(correlation_columns))],
            index=correlation_columns,
            columns=correlation_columns
        )
    )
    .stack()
)

strongest_pair = correlation_pairs.abs().idxmax()
strongest_value = correlation_matrix.loc[
    strongest_pair[0], strongest_pair[1]
]

print("\nStrongest correlation pair:")
print(strongest_pair)
print("Correlation value:", round(strongest_value, 3))

import pandas as pd

# Load raw datasets
customers = pd.read_csv("data/raw/customers.csv")
orders = pd.read_csv("data/raw/orders.csv")

# Convert selected columns temporarily for checking
amount_numeric = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

delivery_numeric = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

age_numeric = pd.to_numeric(
    customers["age"], errors="coerce"
)

# -----------------------------
# ORDER AUDIT
# -----------------------------

print("\n========== ORDER AUDIT ==========")

print("Total order rows:", len(orders))

print("Exact duplicate rows:", orders.duplicated().sum())

print(
    "Pending rows:",
    orders["status"].str.strip().str.lower().eq("pending").sum()
)

print("Distinct order IDs:", orders["order_id"].nunique())

print("Non-numeric or missing amounts:", amount_numeric.isna().sum())

print("Negative delivery times:", (delivery_numeric < 0).sum())

print("Blank delivery times:", delivery_numeric.isna().sum())


# -----------------------------
# CUSTOMER AUDIT
# -----------------------------

print("\n========== CUSTOMER AUDIT ==========")

print("Blank ages:", age_numeric.isna().sum())

print(
    "Ages outside 10-100:",
    ((age_numeric < 10) | (age_numeric > 100)).sum()
)

print(
    "Distinct raw city spellings:",
    customers["city"].nunique()
)

gender = customers["gender"].fillna("").astype(str).str.strip()

print(
    "Distinct non-blank raw gender spellings:",
    gender[gender != ""].nunique()
)

print("Blank genders:", (gender == "").sum())

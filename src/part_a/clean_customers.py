
import pandas as pd

# Step 1: Load the raw customer dataset
df = pd.read_csv("data/raw/customers.csv")

print("Original customer shape:", df.shape)

# Step 2: Standardise city names
df["city"] = df["city"].str.strip().str.title()

# Step 3: Standardise gender values
gender_mapping = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female"
}

df["gender"] = (
    df["gender"]
    .fillna("")
    .str.strip()
    .str.lower()
    .map(gender_mapping)
    .fillna("unknown")
)

# Step 4: Convert age to numeric
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# Step 5: Treat invalid ages as missing
df.loc[
    (df["age"] < 10) | (df["age"] > 100),
    "age"
] = pd.NA

# Step 6: Calculate city-wise median age
city_median_age = df.groupby("city")["age"].transform("median")

# Step 7: Fill missing ages using city median
df["age"] = df["age"].fillna(city_median_age)

# Step 8: Convert signup date to datetime
df["signup_date"] = pd.to_datetime(df["signup_date"])

# Step 9: Save cleaned customer dataset
df.to_csv("data/clean/customers.csv", index=False)

# Step 10: Display validation results
print("\n========== CUSTOMER CLEANING RESULTS ==========")

print("Final shape:", df.shape)

print("\nCity counts:")
print(df["city"].value_counts())

print("\nGender counts:")
print(df["gender"].value_counts())

print("\nMissing ages:", df["age"].isna().sum())

print("Minimum age:", df["age"].min())
print("Maximum age:", df["age"].max())

print("\nSignup date data type:", df["signup_date"].dtype)

print("\nCleaned customer dataset saved successfully!")

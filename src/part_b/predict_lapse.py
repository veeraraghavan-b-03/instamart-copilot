import pandas as pd

# Load cleaned data
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Project cutoff date
cutoff = pd.Timestamp("2025-09-01")

# Consider only completed orders
completed = orders[orders["status"] == "completed"].copy()

# Orders before cutoff
before_cutoff = completed[
    completed["order_ts"] < cutoff
].copy()

# Orders during September
september_orders = completed[
    (completed["order_ts"] >= cutoff) &
    (completed["order_ts"] < pd.Timestamp("2025-10-01"))
].copy()

# Customers who purchased before cutoff
cohort = before_cutoff["customer_id"].unique()

# Determine lapse label
september_customers = set(september_orders["customer_id"])

labels = pd.DataFrame({"customer_id": cohort})

labels["lapsed"] = labels["customer_id"].apply(
    lambda customer_id: int(customer_id not in september_customers)
)

print("========== LAPSE COHORT ==========")
print("Total customers:", len(labels))
print("Lapsed customers:", labels["lapsed"].sum())
print("Active customers:", (labels["lapsed"] == 0).sum())
print("Lapse percentage:", round(labels["lapsed"].mean() * 100, 2), "%")

# ==========================================
# BUILD CUSTOMER FEATURES (BEFORE CUTOFF)
# ==========================================

products = pd.read_csv("data/clean/products.csv")

# All orders before cutoff, including cancellations
all_before = orders[orders["order_ts"] < cutoff].copy()

# Completed orders before cutoff
completed_before = all_before[
    all_before["status"] == "completed"
].copy()

# 1. Recency, frequency and monetary
rfm_features = completed_before.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

rfm_features["recency"] = (
    cutoff - rfm_features["last_order"]
).dt.days

# 2. Average discount
discount_features = completed_before.groupby(
    "customer_id"
).agg(
    avg_discount=("discount_pct", "mean")
).reset_index()

# 3. Average delivery time
delivery_features = completed_before.groupby(
    "customer_id"
).agg(
    avg_delivery=("delivery_minutes", "mean")
).reset_index()

# 4. Cancellation rate
cancellation_features = all_before.groupby(
    "customer_id"
).agg(
    total_orders=("order_id", "nunique"),
    cancelled_orders=(
        "status",
        lambda x: (x == "cancelled").sum()
    )
).reset_index()

cancellation_features["cancellation_rate"] = (
    cancellation_features["cancelled_orders"] /
    cancellation_features["total_orders"]
)

# 5. Most frequently purchased product category
category_data = completed_before.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_features = (
    category_data.groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_count")
    .sort_values(
        ["customer_id", "category_count", "category"],
        ascending=[True, False, True]
    )
    .drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "most_frequent_category"})
)

# 6. Customer demographics and tenure
customer_features = customers[
    ["customer_id", "city", "age", "gender", "signup_date"]
].copy()

customer_features["tenure_days"] = (
    cutoff - customer_features["signup_date"]
).dt.days

customer_features = customer_features.drop(
    columns=["signup_date"]
)

# 7. Combine all features
features = labels.merge(
    rfm_features.drop(columns=["last_order"]),
    on="customer_id",
    how="left"
)

features = features.merge(
    discount_features,
    on="customer_id",
    how="left"
)

features = features.merge(
    delivery_features,
    on="customer_id",
    how="left"
)

features = features.merge(
    cancellation_features[
        ["customer_id", "cancellation_rate"]
    ],
    on="customer_id",
    how="left"
)

features = features.merge(
    category_features,
    on="customer_id",
    how="left"
)

features = features.merge(
    customer_features,
    on="customer_id",
    how="left"
)

# 8. Check the feature dataset
print("\n========== FEATURE DATASET ==========")
print("Shape:", features.shape)

print("\nColumns:")
print(features.columns.tolist())

print("\nMissing values:")
print(features.isnull().sum())

print("\nLapse distribution:")
print(features["lapsed"].value_counts())

print("\nFirst five rows:")
print(features.head())

# Save feature dataset
features.to_csv(
    "outputs/lapse_features.csv",
    index=False
)

print("\nSaved: outputs/lapse_features.csv")


# ==========================================
# MODEL TRAINING AND EVALUATION
# ==========================================

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)

# 1. Separate input features (X) and target (y)
X = features.drop(columns=["customer_id", "lapsed"])
y = features["lapsed"]

numeric_features = [
    "frequency",
    "monetary",
    "recency",
    "avg_discount",
    "avg_delivery",
    "cancellation_rate",
    "age",
    "tenure_days"
]

categorical_features = [
    "most_frequent_category",
    "city",
    "gender"
]

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 3. Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        )
    ]
)

# 4. Evaluation helper function
def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0
    )

    print(f"\n========== {name} ==========")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Lapsed precision:", round(precision, 3))
    print("Lapsed recall:", round(recall, 3))
    print("Lapsed F1-score:", round(f1, 3))

    return predictions

# 5. Baseline model: predict everyone as active
baseline_predictions = [0] * len(y_test)

print("\n========== BASELINE MODEL ==========")
print("Accuracy:", round(
    accuracy_score(y_test, baseline_predictions), 3
))
print("Confusion matrix:")
print(confusion_matrix(y_test, baseline_predictions))

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    baseline_predictions,
    average="binary",
    zero_division=0
)

print("Lapsed precision:", round(precision, 3))
print("Lapsed recall:", round(recall, 3))
print("Lapsed F1-score:", round(f1, 3))

# 6. Logistic Regression pipeline
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

lr_grid = GridSearchCV(
    lr_pipeline,
    {
        "model__C": [0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)

print("\n========== LOGISTIC REGRESSION ==========")
print("Best parameters:", lr_grid.best_params_)
print("Best CV F1:", round(lr_grid.best_score_, 3))

lr_predictions = evaluate_model(
    "LOGISTIC REGRESSION TEST RESULTS",
    lr_grid.best_estimator_,
    X_test,
    y_test
)

# 7. Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_grid = GridSearchCV(
    rf_pipeline,
    {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("\n========== RANDOM FOREST ==========")
print("Best parameters:", rf_grid.best_params_)
print("Best CV F1:", round(rf_grid.best_score_, 3))

rf_predictions = evaluate_model(
    "RANDOM FOREST TEST RESULTS",
    rf_grid.best_estimator_,
    X_test,
    y_test
)



# ==========================================
# OVERSAMPLING THE TRAINING DATA
# ==========================================

from sklearn.base import clone

# Separate active and lapsed training customers
X_active = X_train[y_train == 0]
X_lapsed = X_train[y_train == 1]

y_active = y_train[y_train == 0]
y_lapsed = y_train[y_train == 1]

# Increase lapsed rows to match active rows
X_lapsed_oversampled = X_lapsed.sample(
    n=len(X_active),
    replace=True,
    random_state=42
)

y_lapsed_oversampled = pd.Series(
    1,
    index=X_lapsed_oversampled.index
)

# Combine active and oversampled lapsed customers
X_train_balanced = pd.concat([
    X_active,
    X_lapsed_oversampled
])

y_train_balanced = pd.concat([
    y_active,
    y_lapsed_oversampled
])

print("\n========== OVERSAMPLING RESULTS ==========")
print("Before oversampling:")
print(y_train.value_counts())

print("\nAfter oversampling:")
print(y_train_balanced.value_counts())

# Train a fresh copy of the best Logistic Regression pipeline
balanced_lr = clone(lr_grid.best_estimator_)

balanced_lr.fit(X_train_balanced, y_train_balanced)

# Evaluate on the ORIGINAL test data
balanced_predictions = evaluate_model(
    "OVERSAMPLED LOGISTIC REGRESSION",
    balanced_lr,
    X_test,
    y_test
)

# Compare the two Logistic Regression models
print("\n========== LOGISTIC REGRESSION COMPARISON ==========")

for name, predictions in [
    ("Original", lr_predictions),
    ("Oversampled", balanced_predictions)
]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0
    )

    print(
        f"{name}: Precision={precision:.3f}, "
        f"Recall={recall:.3f}, F1={f1:.3f}"
    )

    # ==========================================
# FEATURE IMPORTANCE AND MODEL INTERPRETATION
# ==========================================

# 1. Random Forest: Top 5 important features
rf_model = rf_grid.best_estimator_

rf_feature_names = rf_model.named_steps[
    "preprocessor"
].get_feature_names_out()

rf_importances = rf_model.named_steps[
    "model"
].feature_importances_

importance_df = pd.DataFrame({
    "feature": rf_feature_names,
    "importance": rf_importances
}).sort_values("importance", ascending=False)

print("\n========== TOP 5 RANDOM FOREST FEATURES ==========")
print(importance_df.head(5).to_string(index=False))


# 2. Logistic Regression: Top 3 coefficients by absolute value
lr_model = lr_grid.best_estimator_

lr_feature_names = lr_model.named_steps[
    "preprocessor"
].get_feature_names_out()

lr_coefficients = lr_model.named_steps[
    "model"
].coef_[0]

coefficient_df = pd.DataFrame({
    "feature": lr_feature_names,
    "coefficient": lr_coefficients
})

coefficient_df["absolute_value"] = (
    coefficient_df["coefficient"].abs()
)

coefficient_df = coefficient_df.sort_values(
    "absolute_value",
    ascending=False
)

print("\n========== TOP 3 LOGISTIC REGRESSION COEFFICIENTS ==========")
print(
    coefficient_df[
        ["feature", "coefficient"]
    ].head(3).to_string(index=False)
)

# ==========================================
# SAVE AND RELOAD FINAL MODEL
# ==========================================

import joblib

joblib.dump(
    balanced_lr,
    "outputs/lapse_model.joblib"
)

print("\nModel saved: outputs/lapse_model.joblib")

# Reload model
loaded_model = joblib.load(
    "outputs/lapse_model.joblib"
)

# Verify predictions are identical
original_predictions = balanced_lr.predict(X_test)
reloaded_predictions = loaded_model.predict(X_test)

print(
    "Reload verification:",
    (original_predictions == reloaded_predictions).all()
)

# ==========================================
# PREMIUM CUSTOMER RETENTION LIST
# ==========================================

# Customer IDs corresponding to the test rows
retention = pd.DataFrame({
    "customer_id": features.loc[
        X_test.index, "customer_id"
    ].values,
    "lapse_probability": loaded_model.predict_proba(
        X_test
    )[:, 1],
    "actually_lapsed": y_test.values
})

# Add premium segment information
segments = pd.read_csv("outputs/customer_segments.csv")

retention = retention.merge(
    segments[["customer_id", "is_premium"]],
    on="customer_id",
    how="left"
)

# Keep premium customers only
premium_retention = retention[
    retention["is_premium"] == True
].copy()

# Rank by predicted lapse probability
premium_retention = premium_retention.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

# Save required columns
retention_output = premium_retention[
    [
        "customer_id",
        "lapse_probability",
        "actually_lapsed"
    ]
]

retention_output.to_csv(
    "outputs/retention_list.csv",
    index=False
)

print("\n========== TOP 10 PREMIUM RETENTION CUSTOMERS ==========")
print(retention_output.to_string(index=False))

print(
    "\nActually lapsed among top 10:",
    int(retention_output["actually_lapsed"].sum())
)

print("\nSaved: outputs/retention_list.csv")
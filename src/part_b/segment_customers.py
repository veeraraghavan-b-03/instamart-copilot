
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# 1. Load cleaned data
# --------------------------------------------------

orders = pd.read_csv("data/clean/orders.csv")
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Keep only completed orders
completed = orders[orders["status"] == "completed"].copy()

# --------------------------------------------------
# 2. Calculate RFM
# --------------------------------------------------

snapshot = pd.Timestamp("2025-10-01 00:00:00")

rfm = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

rfm["recency"] = (
    snapshot - rfm["last_order"]
).dt.days

rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

print("========== RFM SUMMARY ==========")
print("Total customers:", len(rfm))
print("Recency median:", rfm["recency"].median())
print("Recency minimum:", rfm["recency"].min())
print("Recency maximum:", rfm["recency"].max())
print("Frequency mean:", round(rfm["frequency"].mean(), 2))
print("Frequency maximum:", rfm["frequency"].max())
print("Monetary mean:", round(rfm["monetary"].mean(), 2))
print("Monetary maximum:", round(rfm["monetary"].max(), 2))

# --------------------------------------------------
# 3. Standardise RFM values
# --------------------------------------------------

rfm_columns = ["recency", "frequency", "monetary"]

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[rfm_columns])

# --------------------------------------------------
# 4. Test K-means for K = 2 to 8
# --------------------------------------------------

print("\n========== K-MEANS RESULTS ==========")

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(rfm_scaled)

    score = silhouette_score(rfm_scaled, labels)

    print(
        f"K={k} | "
        f"Inertia={model.inertia_:.1f} | "
        f"Silhouette={score:.3f}"
    )



# --------------------------------------------------
# 5. Final clustering with K = 4
# --------------------------------------------------

final_model = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = final_model.fit_predict(rfm_scaled)

# --------------------------------------------------
# 6. Profile each segment
# --------------------------------------------------

segment_profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).round(2)

print("\n========== FOUR SEGMENT PROFILES ==========")
print(segment_profile)

# --------------------------------------------------
# 7. PCA - reduce RFM to two dimensions
# --------------------------------------------------

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(rfm_scaled)

print("\n========== PCA RESULTS ==========")
print(
    "Explained variance ratio:",
    pca.explained_variance_ratio_.round(3)
)
print(
    "Total explained variance:",
    round(pca.explained_variance_ratio_.sum(), 3)
)


# 5. Final clustering using K=4
final_model = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = final_model.fit_predict(rfm_scaled)

# 6. Segment profiles
segment_profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).round(2)

print("\n========== SEGMENT PROFILES ==========")
print(segment_profile)

# 7. PCA analysis
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(rfm_scaled)

print("\n========== PCA RESULTS ==========")
print("Explained variance ratio:", pca.explained_variance_ratio_.round(3))
print("Total variance explained:", round(
    pca.explained_variance_ratio_.sum(), 3
))

# 8. Assign business-friendly segment names

segment_names = {
    0: "Occasional Customers",
    1: "High-Value Customers",
    2: "Loyal Customers",
    3: "At-Risk Customers"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# 9. Identify premium segment using highest average monetary value

premium_segment = segment_profile["mean_monetary"].idxmax()

rfm["is_premium"] = rfm["segment_id"] == premium_segment

# 10. Prepare final output

final_output = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]

# 11. Save CSV

final_output.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

print("\n========== FINAL SEGMENT SUMMARY ==========")
print(final_output["segment_name"].value_counts())

print("\nPremium segment ID:", premium_segment)
print("Premium customers:", int(final_output["is_premium"].sum()))

print("\nSaved: outputs/customer_segments.csv")
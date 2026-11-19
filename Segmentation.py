import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)

print("=" * 60)
print("STAGE 6 - Segmentation and business analysis")
print("=" * 60)
print()

sales = pd.read_csv("sales_enriched.csv")
orders = pd.read_csv("orders_clean.csv")
customers = pd.read_csv("customers_clean.csv")
products = pd.read_csv("products_clean.csv")

sales["OrderDate"] = pd.to_datetime(sales["OrderDate"])
orders["OrderDate"] = pd.to_datetime(orders["OrderDate"])

valid = sales[sales["Status"].isin(["completed", "shipped"])]
valid_orders = orders[orders["Status"].isin(["completed", "shipped"])]

print("--- 2. Customer features ---")
frequency = valid_orders.groupby("CustomerID").size()
total_spend = valid.groupby("CustomerID")["LineTotal"].sum()
total_profit = valid.groupby("CustomerID")["Profit"].sum()

customer_features = pd.DataFrame({
    "CustomerID": customers["CustomerID"],
    "Age": customers["Age"],
    "CustomerCategory": customers["CustomerCategory"],
    "Frequency": [frequency[c] for c in customers["CustomerID"]],
    "TotalSpend": [total_spend[c] for c in customers["CustomerID"]],
    "TotalProfit": [total_profit[c] for c in customers["CustomerID"]],
})
customer_features["AvgOrderValue"] = (
    customer_features["TotalSpend"] / customer_features["Frequency"]
)

print(customer_features.head())
print(customer_features[["Frequency", "TotalSpend", "AvgOrderValue"]].describe())
print()

print("--- 3. Scaling and choosing k ---")
feature_cols = ["Frequency", "TotalSpend", "AvgOrderValue"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_features[feature_cols])

k_values = range(2, 9)
inertias = []
silhouettes = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, model.labels_))
    print("k =", k, "| inertia =", model.inertia_, "| silhouette =", silhouettes[-1])

plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.show()

plt.plot(list(k_values), silhouettes, marker="o")
plt.title("Silhouette score by k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.show()
print()

print("--- 4. K-means ---")
chosen_k = 4

kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)
customer_features["Segment"] = kmeans.labels_

segment_names = {
    0: "Big-basket occasional",
    1: "VIP",
    2: "Low-value occasional",
    3: "Loyal regulars",
}
customer_features["SegmentName"] = [segment_names[s] for s in customer_features["Segment"]]

segment_sizes = customer_features["Segment"].value_counts().sort_index()
print("Customers per segment:")
print(segment_sizes)

profile = customer_features.groupby("Segment")[
    ["Frequency", "TotalSpend", "AvgOrderValue", "TotalProfit", "Age"]
].mean()
print("Average profile of each segment:")
print(profile)

segment_spend = customer_features.groupby("Segment")["TotalSpend"].sum()
segment_share = segment_spend / segment_spend.sum() * 100
print("Share of revenue by segment (%):")
print(segment_share)

print("Segment vs declared runner level:")
print(pd.crosstab(customer_features["Segment"], customer_features["CustomerCategory"]))

plt.figure(figsize=(8, 6))
for seg in range(chosen_k):
    subset = customer_features[customer_features["Segment"] == seg]
    plt.scatter(subset["Frequency"], subset["TotalSpend"], label="Segment " + str(seg))
plt.title("Customer segments")
plt.xlabel("Number of orders")
plt.ylabel("Total spend (EUR)")
plt.legend()
plt.show()

plt.bar(segment_sizes.index, segment_sizes.values)
plt.title("Customers per segment")
plt.xlabel("Segment")
plt.ylabel("Customers")
plt.show()

plt.bar(segment_share.index, segment_share.values)
plt.title("Share of revenue by segment")
plt.xlabel("Segment")
plt.ylabel("Percent of revenue")
plt.show()

top10 = customer_features.sort_values("TotalSpend", ascending=False).head(10)
print("Top 10 customers:")
print(top10[["CustomerID", "Frequency", "TotalSpend", "Segment"]])
print("Revenue share of top 10 customers (%):",
      top10["TotalSpend"].sum() / customer_features["TotalSpend"].sum() * 100)
print()

print("--- 5. Categories ---")
category_table = pd.DataFrame({
    "Revenue": valid.groupby("Category")["LineTotal"].sum(),
    "Cost": valid.groupby("Category")["LineCost"].sum(),
    "Profit": valid.groupby("Category")["Profit"].sum(),
    "Units": valid.groupby("Category")["Quantity"].sum(),
})
category_table["MarginPct"] = category_table["Profit"] / category_table["Revenue"] * 100
category_table["RevenueSharePct"] = category_table["Revenue"] / category_table["Revenue"].sum() * 100
category_table["ProfitSharePct"] = category_table["Profit"] / category_table["Profit"].sum() * 100
print(category_table)

all_revenue = sales.groupby("Category")["LineTotal"].sum()
loss_pct = (1 - category_table["Revenue"] / all_revenue) * 100
print("Revenue lost to cancelled/returned orders by category (%):")
print(loss_pct)

plt.bar(category_table.index, category_table["Revenue"])
plt.title("Revenue by category")
plt.xlabel("Category")
plt.ylabel("Revenue (EUR)")
plt.show()

plt.bar(category_table.index, category_table["Profit"])
plt.title("Profit by category")
plt.xlabel("Category")
plt.ylabel("Profit (EUR)")
plt.show()

plt.bar(category_table.index, category_table["MarginPct"])
plt.title("Profit margin by category")
plt.xlabel("Category")
plt.ylabel("Margin (%)")
plt.show()
print()

print("--- 6. Products ---")
product_table = pd.DataFrame({
    "Revenue": valid.groupby("ProductName")["LineTotal"].sum(),
    "Profit": valid.groupby("ProductName")["Profit"].sum(),
    "Units": valid.groupby("ProductName")["Quantity"].sum(),
})
product_table["MarginPct"] = product_table["Profit"] / product_table["Revenue"] * 100

print("Top 5 products by profit:")
print(product_table.sort_values("Profit", ascending=False).head(5))
print("Bottom 5 products by profit:")
print(product_table.sort_values("Profit").head(5))
print("Top 5 products by margin:")
print(product_table.sort_values("MarginPct", ascending=False).head(5))
print()

print("--- 7. Monthly financials ---")
monthly = valid.set_index("OrderDate")[["LineTotal", "LineCost", "Profit"]].resample("ME").sum()
monthly["MarginPct"] = monthly["Profit"] / monthly["LineTotal"] * 100
print(monthly)

plt.figure(figsize=(12, 5))
plt.plot(monthly.index, monthly["LineTotal"], label="Revenue")
plt.plot(monthly.index, monthly["Profit"], label="Profit")
plt.title("Monthly revenue and profit")
plt.xlabel("Month")
plt.ylabel("EUR")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

last_3 = monthly["LineTotal"].iloc[-3:].sum()
previous_3 = monthly["LineTotal"].iloc[-6:-3].sum()
growth = (last_3 - previous_3) / previous_3 * 100
print("Revenue last 3 months:", last_3)
print("Revenue previous 3 months:", previous_3)
print("Growth (%):", growth)

weakest = monthly["LineTotal"].sort_values().head(3)
print("Weakest months by revenue:")
print(weakest)
print()

print("--- 8. Cities and payments ---")
city_of = {}
for i in range(len(customers)):
    city_of[customers["CustomerID"].iloc[i]] = customers["City"].iloc[i]

by_city = valid.copy()
by_city["City"] = [city_of[c] for c in by_city["CustomerID"]]
print("Revenue by city:")
print(by_city.groupby("City")["LineTotal"].sum().sort_values(ascending=False))

order_totals = valid.groupby("OrderID")["LineTotal"].sum()
payment_orders = valid_orders.copy()
payment_orders["OrderTotal"] = [order_totals[o] for o in payment_orders["OrderID"]]
print("Orders by payment method:")
print(payment_orders.groupby("PaymentMethod")["OrderTotal"].agg(["count", "mean", "sum"]))

print("Average order value:", payment_orders["OrderTotal"].mean())
print()

customer_features.to_csv("customer_segments.csv", index=False)
category_table.to_csv("category_table.csv")
product_table.to_csv("product_table.csv")
monthly.to_csv("monthly_financials.csv")

print("=" * 60)
print("Stage 6 completed. Next: interactive dashboard.")
print("=" * 60)
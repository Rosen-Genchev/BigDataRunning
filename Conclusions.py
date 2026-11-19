import pandas as pd
import matplotlib.pyplot as plt

CUSTOMERS_FILE = "customers_clean.csv"
PRODUCTS_FILE = "products.csv"
ORDERS_FILE = "orders_clean.csv"
DETAILS_FILE = "order_details_clean.csv"
REVIEWS_FILE = "reviews_clean.csv"


print("Conclusions and recommendations")


customers = pd.read_csv(CUSTOMERS_FILE)
products = pd.read_csv(PRODUCTS_FILE)
orders = pd.read_csv(ORDERS_FILE)
details = pd.read_csv(DETAILS_FILE)
reviews = pd.read_csv(REVIEWS_FILE)

orders["OrderDate"] = pd.to_datetime(orders["OrderDate"])

sales = details.merge(orders, on="OrderID").merge(products, on="ProductID")
sales["LineCost"] = sales["Quantity"] * sales["Cost"]
sales["Profit"] = sales["LineTotal"] - sales["LineCost"]

valid = sales[sales["Status"].isin(["completed", "shipped"])]

total_revenue = valid["LineTotal"].sum()
total_profit = valid["Profit"].sum()
print("\nOverall")
print("Revenue:", total_revenue)
print("Profit: ", round(total_profit, 2))
print("Margin %:", round(total_profit / total_revenue * 100, 2))

print("\nCategories")
cat = valid.groupby("Category").agg(
    Revenue=("LineTotal", "sum"),
    Profit=("Profit", "sum")
)
cat["MarginPct"] = cat["Profit"] / cat["Revenue"] * 100
cat["RevenueSharePct"] = cat["Revenue"] / cat["Revenue"].sum() * 100
cat["ProfitSharePct"] = cat["Profit"] / cat["Profit"].sum() * 100
print(cat.round(2))

x = range(len(cat))
plt.figure(figsize=(10, 5))
plt.bar([i - 0.2 for i in x], cat["RevenueSharePct"], width=0.4, label="Revenue share %")
plt.bar([i + 0.2 for i in x], cat["ProfitSharePct"], width=0.4, label="Profit share %")
plt.xticks(list(x), cat.index)
plt.title("Revenue share vs profit share by category")
plt.xlabel("Category")
plt.ylabel("Share (%)")
plt.legend()
plt.tight_layout()
plt.savefig("stage8_category_shares.png")
plt.show()

print("\nProducts: profit vs rating")
prod = valid.groupby("ProductName").agg(
    Revenue=("LineTotal", "sum"),
    Profit=("Profit", "sum"),
    Units=("Quantity", "sum")
)
prod["MarginPct"] = prod["Profit"] / prod["Revenue"] * 100

rating = reviews.merge(products, on="ProductID").groupby("ProductName").agg(
    AvgRating=("Rating", "mean"),
    Reviews=("Rating", "count")
)
prod = prod.join(rating)
print(prod.sort_values("Profit", ascending=False).round(2))

print("\nHigh profit (top 5) but rating below 3.5:")
top5 = prod.sort_values("Profit", ascending=False).head(5)
print(top5[top5["AvgRating"] < 3.5].round(2))

print("\nMonthly revenue")
monthly = valid.set_index("OrderDate")["LineTotal"].resample("ME").sum()
print(monthly)
print("Best months:")
print(monthly.sort_values(ascending=False).head(3))
print("Weakest months:")
print(monthly.sort_values().head(3))

last3 = monthly.iloc[-3:].sum()
prev3 = monthly.iloc[-6:-3].sum()
print("Growth last 3 vs previous 3 (%):", round((last3 - prev3) / prev3 * 100, 2))

plt.figure(figsize=(12, 5))
plt.bar(monthly.index.strftime("%Y-%m"), monthly.values)
plt.title("Monthly revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("stage8_monthly_revenue.png")
plt.show()

print("\nLost revenue")
lost = sales[sales["Status"].isin(["cancelled", "returned"])]["LineTotal"].sum()
gross = sales["LineTotal"].sum()
print("Lost revenue:", lost, "of", gross)
print("Lost %:", round(lost / gross * 100, 2))


print("MAIN CONCLUSIONS")

best_margin = cat["MarginPct"].idxmax()
worst_margin = cat["MarginPct"].idxmin()
print(f"1. Highest-margin category: {best_margin} "
      f"({cat.loc[best_margin, 'MarginPct']:.1f}%)")
print(f"2. Lowest-margin category: {worst_margin} "
      f"({cat.loc[worst_margin, 'MarginPct']:.1f}%)")
print(f"3. Weakest month: {monthly.idxmin().strftime('%Y-%m')} "
      f"({monthly.min():.0f})")
print(f"4. Strongest month: {monthly.idxmax().strftime('%Y-%m')} "
      f"({monthly.max():.0f})")
print(f"5. Revenue lost to cancelled/returned orders: "
      f"{lost / gross * 100:.1f}%")

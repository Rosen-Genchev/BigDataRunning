import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform, binom
from scipy.stats import pearsonr, spearmanr

np.random.seed(42)


print("Statistical analysis")
print()

customers = pd.read_csv("customers_clean.csv")
products = pd.read_csv("products_clean.csv")
orders = pd.read_csv("orders_clean.csv")
details = pd.read_csv("order_details_clean.csv")
reviews = pd.read_csv("reviews_clean.csv")


name_of = {}
category_of = {}
cost_of = {}
for i in range(len(products)):
    pid = products["ProductID"].iloc[i]
    name_of[pid] = products["ProductName"].iloc[i]
    category_of[pid] = products["Category"].iloc[i]
    cost_of[pid] = products["Cost"].iloc[i]

customer_of_order = {}
date_of_order = {}
status_of_order = {}
for i in range(len(orders)):
    oid = orders["OrderID"].iloc[i]
    customer_of_order[oid] = orders["CustomerID"].iloc[i]
    date_of_order[oid] = orders["OrderDate"].iloc[i]
    status_of_order[oid] = orders["Status"].iloc[i]

sales = details.copy()
sales["CustomerID"] = [customer_of_order[o] for o in sales["OrderID"]]
sales["OrderDate"] = [date_of_order[o] for o in sales["OrderID"]]
sales["Status"] = [status_of_order[o] for o in sales["OrderID"]]
sales["ProductName"] = [name_of[p] for p in sales["ProductID"]]
sales["Category"] = [category_of[p] for p in sales["ProductID"]]
sales["UnitCost"] = [cost_of[p] for p in sales["ProductID"]]
sales["OrderDate"] = pd.to_datetime(sales["OrderDate"])
sales["LineCost"] = sales["Quantity"] * sales["UnitCost"]
sales["Profit"] = sales["LineTotal"] - sales["LineCost"]

print(sales.head())
print("Shape:", sales.shape)
print()

sales.to_csv("sales_enriched.csv", index=False)

print("Revenue by order status:")
print(sales.groupby("Status")["LineTotal"].sum())

valid = sales[sales["Status"].isin(["completed", "shipped"])]
print("Lines in valid orders:", len(valid), "of", len(sales))
print()

order_totals = valid.groupby("OrderID")["LineTotal"].sum()
print("Orders with valid revenue:", len(order_totals))
print()


def describe_column(name, column):
    print("", name, "")
    print("Mean:     ", column.mean())
    print("Median:   ", column.median())
    print("Mode:     ", column.mode().tolist()[:3])
    print("Min:      ", column.min())
    print("Max:      ", column.max())
    print("Range:    ", column.max() - column.min())
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    print("Q1:       ", Q1)
    print("Q3:       ", Q3)
    print("IQR:      ", Q3 - Q1)
    print("Variance: ", column.var())
    print("Std:      ", column.std())
    if abs(column.mean() - column.median()) < 0.1 * column.std():
        print("Mean ~ Median: approximately symmetric")
    elif column.mean() > column.median():
        print("Mean > Median: possible right skew")
    else:
        print("Mean < Median: possible left skew")
    print()


print("Descriptive statistics ")
describe_column("Product price (catalog)", products["Price"])
describe_column("Quantity per line", valid["Quantity"])
describe_column("Line total", valid["LineTotal"])
describe_column("Order total", order_totals)

print(valid[["Quantity", "UnitPrice", "LineTotal", "Profit"]].describe())
print()

print("Histograms and box plot ")

plt.hist(products["Price"], bins=10)
plt.title("Distribution of product prices")
plt.xlabel("Price (EUR)")
plt.ylabel("Number of products")
plt.show()

plt.hist(valid["Quantity"], bins=5)
plt.title("Distribution of quantities per order line")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.show()

plt.hist(valid["LineTotal"], bins=30)
plt.title("Distribution of line totals")
plt.xlabel("Line total (EUR)")
plt.ylabel("Frequency")
plt.show()

plt.hist(order_totals, bins=30)
plt.title("Distribution of order totals")
plt.xlabel("Order total (EUR)")
plt.ylabel("Frequency")
plt.show()

plt.boxplot(order_totals)
plt.title("Box plot of order totals")
plt.ylabel("Order total (EUR)")
plt.show()

print("Normal distribution")
mu = order_totals.mean()
sigma = order_totals.std()
print("mu =", mu, "| sigma =", sigma)

simulated = np.random.normal(mu, sigma, 1000)
print("Simulated min:", simulated.min(), "| Real min:", order_totals.min())

plt.hist(simulated, bins=30)
plt.title("Simulated normal distribution with the same mean and std")
plt.xlabel("Order total (EUR)")
plt.ylabel("Frequency")
plt.show()

p_theory = norm.cdf(500, loc=mu, scale=sigma)
p_real = (order_totals <= 500).mean()
print("P(order <= 500): theory =", p_theory, "| real =", p_real)

p_negative = norm.cdf(0, loc=mu, scale=sigma)
print("Normal model says P(order < 0) =", p_negative, "(impossible in reality)")

x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 500)
plt.plot(x, norm.pdf(x, mu, sigma))
plt.axvline(500)
plt.title("Normal model of the order total (line at 500)")
plt.xlabel("Order total (EUR)")
plt.ylabel("Density")
plt.show()

print("Uniform distribution")
plt.hist(customers["Age"], bins=10)
plt.title("Distribution of customer age")
plt.xlabel("Age")
plt.ylabel("Number of customers")
plt.show()

p_uniform = uniform.cdf(50, loc=18, scale=47) - uniform.cdf(30, loc=18, scale=47)
p_age_real = ((customers["Age"] >= 30) & (customers["Age"] <= 50)).mean()
print("P(30 <= Age <= 50): uniform[18, 65] theory =", p_uniform, "| real =", p_age_real)

uniform_sample = np.random.uniform(18, 65, 1000)
print("Simulated uniform mean:", uniform_sample.mean(), "| Real mean age:", customers["Age"].mean())

print("Binomial distribution")
p_problem = orders["Status"].isin(["cancelled", "returned"]).mean()
n = 20
print("Share of cancelled/returned orders p =", p_problem)
print("Expected problem orders in 20:", n * p_problem)

print("P(X = 4):  ", binom.pmf(4, n, p_problem))
print("P(X >= 6): ", 1 - binom.cdf(5, n, p_problem))
print("P(X <= 2): ", binom.cdf(2, n, p_problem))

x_binom = np.arange(0, n + 1)
probabilities = binom.pmf(x_binom, n, p_problem)
print("Sum of probabilities:", probabilities.sum())

plt.bar(x_binom, probabilities)
plt.title("Problem orders among the next 20 orders (binomial)")
plt.xlabel("Number of cancelled/returned orders")
plt.ylabel("Probability")
plt.show()


def describe_r(r):
    size = abs(r)
    if size >= 0.7:
        strength = "strong"
    elif size >= 0.4:
        strength = "moderate"
    else:
        strength = "weak / none"
    direction = "positive" if r > 0 else "negative"
    return strength + " " + direction


print("Correlation: order lines")
line_cols = ["Quantity", "UnitPrice", "UnitCost", "LineTotal", "Profit"]

pearson_matrix = valid[line_cols].corr(method="pearson")
spearman_matrix = valid[line_cols].corr(method="spearman")
print("Pearson:")
print(pearson_matrix)
print("Spearman:")
print(spearman_matrix)

sns.heatmap(pearson_matrix, annot=True, fmt=".2f")
plt.title("Pearson correlation matrix - order lines")
plt.show()

sns.heatmap(spearman_matrix, annot=True, fmt=".2f")
plt.title("Spearman correlation matrix - order lines")
plt.show()

r, p = pearsonr(valid["UnitPrice"], valid["Quantity"])
rho, p_s = spearmanr(valid["UnitPrice"], valid["Quantity"])
print("UnitPrice vs Quantity: Pearson r =", r, "(p =", p, ") ->", describe_r(r))
print("UnitPrice vs Quantity: Spearman rho =", rho, "(p =", p_s, ") ->", describe_r(rho))

plt.scatter(valid["UnitPrice"], valid["Quantity"])
plt.title("Unit price and quantity per line")
plt.xlabel("Unit price (EUR)")
plt.ylabel("Quantity")
plt.show()

print("Correlation: products")
units = valid.groupby("ProductID")["Quantity"].sum()
revenue = valid.groupby("ProductID")["LineTotal"].sum()
profit = valid.groupby("ProductID")["Profit"].sum()

product_stats = pd.DataFrame({
    "Price": products["Price"],
    "UnitsSold": [units[p] for p in products["ProductID"]],
    "Revenue": [revenue[p] for p in products["ProductID"]],
    "Profit": [profit[p] for p in products["ProductID"]],
})
print(product_stats)

print("Pearson:")
print(product_stats.corr(method="pearson"))
print("Spearman:")
print(product_stats.corr(method="spearman"))

r, p = pearsonr(product_stats["Price"], product_stats["UnitsSold"])
rho, p_s = spearmanr(product_stats["Price"], product_stats["UnitsSold"])
print("Price vs UnitsSold: Pearson r =", r, "(p =", p, ") ->", describe_r(r))
print("Price vs UnitsSold: Spearman rho =", rho, "(p =", p_s, ") ->", describe_r(rho))

plt.scatter(product_stats["Price"], product_stats["UnitsSold"])
plt.title("Product price and units sold")
plt.xlabel("Price (EUR)")
plt.ylabel("Units sold")
plt.show()

sns.heatmap(product_stats.corr(method="spearman"), annot=True, fmt=".2f")
plt.title("Spearman correlation matrix - products")
plt.show()

print("10. Correlation: customers ")
spend = valid.groupby("CustomerID")["LineTotal"].sum()
customer_stats = pd.DataFrame({
    "Age": customers["Age"],
    "TotalSpend": [spend[c] for c in customers["CustomerID"]],
})

r, p = pearsonr(customer_stats["Age"], customer_stats["TotalSpend"])
rho, p_s = spearmanr(customer_stats["Age"], customer_stats["TotalSpend"])
print("Age vs TotalSpend: Pearson r =", r, "(p =", p, ") ->", describe_r(r))
print("Age vs TotalSpend: Spearman rho =", rho, "(p =", p_s, ") ->", describe_r(rho))

plt.scatter(customer_stats["Age"], customer_stats["TotalSpend"])
plt.title("Customer age and total spending")
plt.xlabel("Age")
plt.ylabel("Total spending (EUR)")
plt.show()



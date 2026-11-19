import pandas as pd

customers = pd.read_csv("customers.csv")
products = pd.read_csv("products.csv")
orders = pd.read_csv("orders.csv")
details = pd.read_csv("order_details.csv")
reviews = pd.read_csv("reviews.csv")

tables = {
    "customers": customers,
    "products": products,
    "orders": orders,
    "order_details": details,
    "reviews": reviews,
}

for name, df in tables.items():
    print(name.upper())
    print("Shape:", df.shape)
    print(df.head())
    print(df.dtypes)
    df.info()

print(customers.describe())
print(products.describe())
print(details.describe())
print(reviews.describe())

for name, df in tables.items():
    print(name, "missing values:")
    print(df.isnull().sum())
    print(name, "duplicates:", df.duplicated().sum())

print("Orders with unknown customer:",
      (~orders["CustomerID"].isin(customers["CustomerID"])).sum())
print("Order lines with unknown order:",
      (~details["OrderID"].isin(orders["OrderID"])).sum())
print("Order lines with unknown product:",
      (~details["ProductID"].isin(products["ProductID"])).sum())
print("Reviews with unknown order:",
      (~reviews["OrderID"].isin(orders["OrderID"])).sum())

mismatch = details["Quantity"] * details["UnitPrice"] != details["LineTotal"]
print("Lines where Quantity * UnitPrice != LineTotal:", mismatch.sum())

print(customers["City"].value_counts())
print(orders["Status"].value_counts())
print(orders["PaymentMethod"].value_counts())
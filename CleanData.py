import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer


print("Preprocessing")
print()

customers = pd.read_csv("customers.csv")
products = pd.read_csv("products.csv")
orders = pd.read_csv("orders.csv")
details = pd.read_csv("order_details.csv")
reviews = pd.read_csv("reviews.csv")

before_shapes = {
    "customers": customers.shape,
    "orders": orders.shape,
    "order_details": details.shape,
    "reviews": reviews.shape,
}
print(before_shapes)
print()

print("Quality overview")
customers.info()
details.info()
reviews.info()

print("\nCount of missing values:")
print(customers.isna().sum())
print(details.isna().sum())
print(reviews.isna().sum())

print("\nPercent of missing values:")
print(customers.isna().mean() * 100)
print(details.isna().mean() * 100)
print(reviews.isna().mean() * 100)
print()

print("Standardizing text")
print("City BEFORE:")
print(customers["City"].value_counts())

customers["City"] = customers["City"].str.strip().str.lower()
orders["Status"] = orders["Status"].str.strip().str.lower()
orders["PaymentMethod"] = orders["PaymentMethod"].str.strip().str.lower()

print("\nCity AFTER:")
print(customers["City"].value_counts())
print(orders["Status"].value_counts())
print(orders["PaymentMethod"].value_counts())
print()

print("Converting dates")
customers["RegistrationDate"] = pd.to_datetime(customers["RegistrationDate"])
orders["OrderDate"] = pd.to_datetime(orders["OrderDate"])
reviews["ReviewDate"] = pd.to_datetime(reviews["ReviewDate"])

print(customers.dtypes)
print(orders.dtypes)
print(reviews.dtypes)
print()

print("Duplicates")
print("Duplicate orders:       ", orders.duplicated().sum())
print("Duplicate order lines:  ", details.duplicated().sum())
print("Duplicate reviews:      ", reviews.duplicated().sum())

orders = orders.drop_duplicates()
details = details.drop_duplicates()
reviews = reviews.drop_duplicates()

print("After drop_duplicates:")
print("Orders:", orders.shape, "| Lines:", details.shape, "| Reviews:", reviews.shape)
print()

print("Missing values")

reviews = reviews.dropna(subset=["ReviewText"])

details["Quantity"] = details["Quantity"].fillna(details["LineTotal"] / details["UnitPrice"])
details["UnitPrice"] = details["UnitPrice"].fillna(details["LineTotal"] / details["Quantity"])

details["Quantity"] = details["Quantity"].fillna(details["Quantity"].mean())
details["UnitPrice"] = details["UnitPrice"].fillna(details["UnitPrice"].mean())

print("Missing after cleaning:")
print(details.isna().sum())
print(reviews.isna().sum())

mismatch = details["Quantity"] * details["UnitPrice"] != details["LineTotal"]
print("Lines where Quantity * UnitPrice != LineTotal:", mismatch.sum())
print()

print("Age")
Q1 = customers["Age"].quantile(0.25)
Q3 = customers["Age"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("Q1:", Q1, "| Q3:", Q3, "| IQR:", IQR)
print("Lower bound:", lower, "| Upper bound:", upper)

age_outliers = customers[
    (customers["Age"] < lower) |
    (customers["Age"] > upper)
]
print("Age outliers:")
print(age_outliers[["CustomerID", "Age"]])

customers["Age"] = customers["Age"].clip(lower=lower, upper=upper)

median_age = customers["Age"].median()
customers["Age"] = customers["Age"].fillna(median_age)

print("Age after cleaning:")
print(customers["Age"].describe())
print()

print("Quantity")
print("Statistics BEFORE:")
print(details["Quantity"].describe())

plt.boxplot(details["Quantity"])
plt.title("Quantity - before outlier removal")
plt.ylabel("Quantity")
plt.show()

Q1 = details["Quantity"].quantile(0.25)
Q3 = details["Quantity"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("Q1:", Q1, "| Q3:", Q3, "| IQR:", IQR)
print("Lower bound:", lower, "| Upper bound:", upper)

iqr_mask = (details["Quantity"] < lower) | (details["Quantity"] > upper)
print("IQR outliers:", iqr_mask.sum())

details["Quantity_Z"] = zscore(details["Quantity"])
z_mask = details["Quantity_Z"].abs() > 3
print("Z-score outliers (|Z| > 3):", z_mask.sum())
print("Flagged by both methods:", (iqr_mask & z_mask).sum())

print("Outlier rows:")
print(details[iqr_mask])

details_clean = details[~iqr_mask][["OrderID", "ProductID", "Quantity", "UnitPrice", "LineTotal"]]

print("Statistics AFTER:")
print(details_clean["Quantity"].describe())

plt.boxplot(details_clean["Quantity"])
plt.title("Quantity - after outlier removal")
plt.ylabel("Quantity")
plt.show()
print()

print("LineTotal")
Q1 = details_clean["LineTotal"].quantile(0.25)
Q3 = details_clean["LineTotal"].quantile(0.75)
IQR = Q3 - Q1
upper_total = Q3 + 1.5 * IQR

high_lines = details_clean[details_clean["LineTotal"] > upper_total]
print("Upper IQR bound:", upper_total)
print("Lines above it:", len(high_lines), "(expensive items - real values, kept)")
print()

print("Before / after")
print("customers:    ", before_shapes["customers"], "->", customers.shape)
print("orders:       ", before_shapes["orders"], "->", orders.shape)
print("order_details:", before_shapes["order_details"], "->", details_clean.shape)
print("reviews:      ", before_shapes["reviews"], "->", reviews.shape)

print("\nQuantity BEFORE:")
print(details["Quantity"].describe())
print("Quantity AFTER:")
print(details_clean["Quantity"].describe())

print("\nLineTotal BEFORE:")
print(details["LineTotal"].describe())
print("LineTotal AFTER:")
print(details_clean["LineTotal"].describe())

print("\nTotal revenue BEFORE:", details["LineTotal"].sum())
print("Total revenue AFTER: ", details_clean["LineTotal"].sum())
print()

print("Saving")
customers.to_csv("customers_clean.csv", index=False)
products.to_csv("products_clean.csv", index=False)
orders.to_csv("orders_clean.csv", index=False)
details_clean.to_csv("order_details_clean.csv", index=False)
reviews.to_csv("reviews_clean.csv", index=False)
print("Cleaned files saved.")
print()

print("Encoding")

le = LabelEncoder()
customers["CustomerCategory_Label"] = le.fit_transform(customers["CustomerCategory"])
print("Classes:", list(le.classes_))
print(customers[["CustomerCategory", "CustomerCategory_Label"]].head(10))

customers_onehot = pd.get_dummies(customers, columns=["City", "CustomerCategory"])
print(customers_onehot.head())
print(list(customers_onehot.columns))

X_customers = customers[["City", "CustomerCategory", "Age"]]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["City", "CustomerCategory"]),
        ("num", StandardScaler(), ["Age"])
    ]
)
X_customers_processed = preprocessor.fit_transform(X_customers)
print("X_customers_processed:", X_customers_processed.shape)
print(X_customers_processed[:5])
print()

print("Scaling")
details_mm = details_clean[["Quantity", "UnitPrice", "LineTotal"]].copy()
mm = MinMaxScaler()
details_mm[["Quantity", "UnitPrice", "LineTotal"]] = mm.fit_transform(details_mm)
print("Min-Max:")
print(details_mm.describe())

details_std = details_clean[["Quantity", "UnitPrice", "LineTotal"]].copy()
ss = StandardScaler()
details_std[["Quantity", "UnitPrice", "LineTotal"]] = ss.fit_transform(details_std)
print("Standardization:")
print(details_std.describe())


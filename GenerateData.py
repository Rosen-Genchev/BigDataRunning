import pandas as pd
import numpy as np
from faker import Faker
from datetime import date

Faker.seed(42)
np.random.seed(42)
fake = Faker("bg_BG")

CITIES = ("София", "Пловдив", "Варна", "Бургас", "Русе")
CATEGORIES_CUSTOMER = ("Beginner", "Recreational", "Competitive")
STATUSES = ("Completed",) * 8 + ("Shipped",) * 2 + ("Cancelled",) + ("Returned",)
PAYMENTS = ("Card", "Cash on delivery", "PayPal", "Bank transfer")


customers = []
for i in range(1, 101):
    customers.append({
        "CustomerID": f"C{i:03d}",
        "Name": fake.name(),
        "Age": fake.random_int(min=18, max=65),
        "City": fake.random_element(elements=CITIES),
        "RegistrationDate": fake.date_between(start_date=date(2026, 1, 1), end_date=date(2027, 1, 1)),
        "CustomerCategory": fake.random_element(elements=CATEGORIES_CUSTOMER),
    })


for i in range(0, 100, 10):
    customers[i]["City"] = customers[i]["City"].lower()
for i in range(3, 100, 25):
    customers[i]["City"] = "  " + customers[i]["City"] + " "
customers[7]["Age"] = None
customers[44]["Age"] = None
customers[61]["Age"] = 250

product_list = [
    ("Road Running Shoes", "Footwear", 220, 350),
    ("Trail Running Shoes", "Footwear", 240, 380),
    ("Racing Flats", "Footwear", 280, 450),
    ("Running Shorts", "Apparel", 40, 90),
    ("Technical T-Shirt", "Apparel", 35, 80),
    ("Wind Jacket", "Apparel", 120, 250),
    ("Compression Socks", "Apparel", 20, 45),
    ("GPS Running Watch", "Accessories", 350, 1200),
    ("Hydration Belt", "Accessories", 40, 90),
    ("Running Headlamp", "Accessories", 50, 150),
    ("Foam Roller", "Accessories", 30, 70),
    ("Running Cap", "Accessories", 20, 45),
    ("Energy Gels Box", "Nutrition", 30, 50),
    ("Electrolyte Tablets", "Nutrition", 15, 30),
    ("Recovery Drink Powder", "Nutrition", 45, 80),
    ("Energy Bars Pack", "Nutrition", 25, 45),
    ("Whey Protein", "Supplements", 60, 120),
    ("Creatine Monohydrate", "Supplements", 40, 80),
    ("Magnesium Complex", "Supplements", 20, 45),
    ("Omega-3 Capsules", "Supplements", 30, 60),
]


cost_share = {
    "Footwear": (0.55, 0.75),
    "Apparel": (0.45, 0.65),
    "Accessories": (0.55, 0.75),
    "Nutrition": (0.40, 0.60),
    "Supplements": (0.35, 0.55),
}


max_quantity = {
    "Footwear": 2,
    "Apparel": 3,
    "Accessories": 2,
    "Nutrition": 5,
    "Supplements": 3,
}

products = []
for i in range(len(product_list)):
    name, category, low, high = product_list[i]
    price = fake.random_int(min=low, max=high)
    share_low, share_high = cost_share[category]
    cost = round(price * np.random.uniform(share_low, share_high), 2)
    products.append({
        "ProductID": f"P{i:02d}",
        "ProductName": name,
        "Category": category,
        "Price": price,
        "Cost": cost,
    })


price_of = {}
name_of = {}
category_of = {}
for p in products:
    price_of[p["ProductID"]] = p["Price"]
    name_of[p["ProductID"]] = p["ProductName"]
    category_of[p["ProductID"]] = p["Category"]


month_weight = {1: 0.7, 2: 0.7, 3: 1.1, 4: 1.5, 5: 1.6, 6: 1.0,
                7: 0.8, 8: 0.8, 9: 1.3, 10: 1.5, 11: 1.2, 12: 1.1}

orders = []
details = []
order_customer = {}
order_date_of = {}

n = 0
while len(orders) < 1100:
    order_date = fake.date_between(start_date=date(2026, 1, 1), end_date=date(2027, 1, 1))
    if np.random.uniform(0, 1) > month_weight[order_date.month] / 1.6:
        continue
    n += 1
    order_id = f"O{n:05d}"
    customer_id = f"C{fake.random_int(min=1, max=100):03d}"
    orders.append({
        "OrderID": order_id,
        "CustomerID": customer_id,
        "OrderDate": order_date,
        "Status": fake.random_element(elements=STATUSES),
        "PaymentMethod": fake.random_element(elements=PAYMENTS),
    })
    order_customer[order_id] = customer_id
    order_date_of[order_id] = order_date

    used_products = []
    for _ in range(fake.random_int(min=1, max=4)):
        product_id = f"P{fake.random_int(min=0, max=19):02d}"
        while product_id in used_products:
            product_id = f"P{fake.random_int(min=0, max=19):02d}"
        used_products.append(product_id)
        quantity = fake.random_int(min=1, max=max_quantity[category_of[product_id]])
        unit_price = price_of[product_id]
        details.append({
            "OrderID": order_id,
            "ProductID": product_id,
            "Quantity": quantity,
            "UnitPrice": unit_price,
            "LineTotal": quantity * unit_price,
        })

orders.append(orders[5])
orders.append(orders[250])
orders.append(orders[800])
orders[10]["Status"] = "completed"
orders[20]["Status"] = "COMPLETED "
orders[30]["PaymentMethod"] = "card"
orders[40]["PaymentMethod"] = "  Card"

nutrition_lines = []
for idx in range(len(details)):
    if category_of[details[idx]["ProductID"]] == "Nutrition":
        nutrition_lines.append(idx)

details.append(details[3])
details.append(details[100])
details[15]["Quantity"] = None
details[60]["Quantity"] = None
details[200]["UnitPrice"] = None
for idx in (nutrition_lines[5], nutrition_lines[100], nutrition_lines[300]):
    details[idx]["Quantity"] = 60
    details[idx]["LineTotal"] = 60 * details[idx]["UnitPrice"]

positive = ["Great {p}, perfect for my long runs", "Excellent quality, very happy with the {p}",
            "Amazing {p}, I love it", "Fast delivery and a wonderful {p}"]
neutral = ["The {p} is okay", "Average {p}, nothing special",
           "The {p} arrived on time", "Decent {p} for the price"]
negative = ["Terrible {p}, very disappointed", "The {p} was uncomfortable and poor quality",
            "Awful {p}, do not buy", "Bad service and a broken {p}"]

reviews = []
for r in range(1, 321):
    line = details[fake.random_int(min=0, max=len(details) - 1)]
    order_id = line["OrderID"]
    product_id = line["ProductID"]
    product_name = name_of[product_id]

    rating = fake.random_element(elements=(1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5))
    if rating >= 4:
        text = fake.random_element(elements=positive)
    elif rating == 3:
        text = fake.random_element(elements=neutral)
    else:
        text = fake.random_element(elements=negative)
    text = text.format(p=product_name)

    if r % 7 == 0:
        text = text + " https://example.com/review" + str(r)
    if r % 9 == 0:
        text = text + " @run_support"
    if r % 11 == 0:
        text = text + " #" + product_name.replace(" ", "").lower()
    if r % 13 == 0:
        text = text.upper() + "!!!"

    reviews.append({
        "ReviewID": f"R{r:04d}",
        "OrderID": order_id,
        "CustomerID": order_customer[order_id],
        "ProductID": product_id,
        "ReviewDate": fake.date_between(start_date=order_date_of[order_id], end_date=date(2026, 12, 31)),
        "ReviewText": text,
        "Rating": rating,
    })

reviews.append(reviews[10])
reviews.append(reviews[200])
reviews[25]["ReviewText"] = None
reviews[150]["ReviewText"] = None

pd.DataFrame(customers).to_csv("customers.csv", index=False)
pd.DataFrame(products).to_csv("products.csv", index=False)
pd.DataFrame(orders).to_csv("orders.csv", index=False)
pd.DataFrame(details).to_csv("order_details.csv", index=False)
pd.DataFrame(reviews).to_csv("reviews.csv", index=False)

print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))
print("Order lines:", len(details))
print("Reviews:", len(reviews))
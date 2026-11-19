import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

np.random.seed(42)


print("Time series")

print()


sales = pd.read_csv("sales_enriched.csv")
orders = pd.read_csv("orders_clean.csv")

sales["OrderDate"] = pd.to_datetime(sales["OrderDate"])
orders["OrderDate"] = pd.to_datetime(orders["OrderDate"])

valid_sales = sales[sales["Status"].isin(["completed", "shipped"])]
valid_orders = orders[orders["Status"].isin(["completed", "shipped"])]
print("Valid lines:", len(valid_sales), "| Valid orders:", len(valid_orders))
print()

print("Daily revenue and orders")
daily_revenue = valid_sales.groupby("OrderDate")["LineTotal"].sum()
daily_orders = valid_orders.groupby("OrderDate").size()

df = pd.DataFrame({"Revenue": daily_revenue, "Orders": daily_orders})
df.index.name = "Date"
print(df.head())
print("Days with orders:", len(df))
print()

print("Time index")
print("Start:", df.index.min())
print("End:  ", df.index.max())
print("Monotonic increasing:", df.index.is_monotonic_increasing)

df = df.sort_index()
print("Missing values:")
print(df.isna().sum())
print("Intervals between observations:")
print(df.index.to_series().diff().value_counts())
print()

print("Full daily index")
full_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="D"
)
df = df.reindex(full_index)
df.index.name = "Date"

print("Days without orders:", df["Revenue"].isna().sum(), "of", len(df))

df = df.fillna(0)
print(df.isna().sum())
print()

print("Daily revenue")
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Revenue"], label="Daily revenue")
plt.plot(df.index, df["Revenue"].rolling(7).mean(), label="7-day average")
plt.plot(df.index, df["Revenue"].rolling(30).mean(), label="30-day average")
plt.title("Daily revenue")
plt.xlabel("Date")
plt.ylabel("Revenue (EUR)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Seasonality")
df["DayOfWeek"] = df.index.day_of_week
weekday_revenue = df.groupby("DayOfWeek")["Revenue"].mean()
print("Average revenue by day of week (0 = Monday):")
print(weekday_revenue)

plt.bar(weekday_revenue.index, weekday_revenue.values)
plt.title("Average revenue by day of week")
plt.xlabel("Day of week (0 = Monday)")
plt.ylabel("Average revenue (EUR)")
plt.show()

monthly_revenue = df["Revenue"].resample("ME").sum()
monthly_orders = df["Orders"].resample("ME").sum()
print("Monthly revenue:")
print(monthly_revenue)
print("Monthly orders:")
print(monthly_orders)

plt.figure(figsize=(12, 5))
plt.plot(monthly_revenue.index, monthly_revenue.values)
plt.title("Monthly revenue")
plt.xlabel("Month")
plt.ylabel("Revenue (EUR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(monthly_orders.index, monthly_orders.values)
plt.title("Monthly number of orders")
plt.xlabel("Month")
plt.ylabel("Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

weekly_revenue = df["Revenue"].resample("W").sum()
weekly_orders = df["Orders"].resample("W").sum()

plt.figure(figsize=(12, 5))
plt.plot(weekly_revenue.index, weekly_revenue.values)
plt.title("Weekly revenue")
plt.xlabel("Week")
plt.ylabel("Revenue (EUR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(weekly_orders.index, weekly_orders.values)
plt.title("Weekly number of orders")
plt.xlabel("Week")
plt.ylabel("Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Train/Test split")
split_date = "2026-09-01"

train = df.loc[df.index < split_date]
test = df.loc[df.index >= split_date]

if len(test) == 0 or len(train) < 60:
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    print("Warning: automatic 80/20 split, the date is outside the data range.")

print("Train:", train.shape, "from", train.index.min(), "to", train.index.max())
print("Test: ", test.shape, "from", test.index.min(), "to", test.index.max())

plt.figure(figsize=(12, 5))
plt.plot(train.index, train["Revenue"], label="Train")
plt.plot(test.index, test["Revenue"], label="Test")
plt.title("Train/Test split")
plt.xlabel("Date")
plt.ylabel("Revenue (EUR)")
plt.legend()
plt.show()

print("Baseline forecasts")

last_value = train["Revenue"].iloc[-1]
naive_forecast = np.repeat(last_value, len(test))
print("Naive - last value in train:", last_value)

window = 7
moving_average = train["Revenue"].rolling(window).mean().iloc[-1]
ma_forecast = np.repeat(moving_average, len(test))
print("Moving average (7 days):", moving_average)

X_train = np.arange(len(train)).reshape(-1, 1)
y_train = train["Revenue"].values
X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
linear_forecast = model_lr.predict(X_test)

print("Intercept:", model_lr.intercept_)
print("Slope:    ", model_lr.coef_[0])
print()

actual_test = test["Revenue"].values


def evaluate(name, actual, prediction):
    mae = mean_absolute_error(actual, prediction)
    mse = mean_squared_error(actual, prediction)
    rmse = np.sqrt(mse)
    print(name)
    print("  MAE: ", mae)
    print("  MSE: ", mse)
    print("  RMSE:", rmse)
    return mae, mse, rmse


model_names = []
maes = []
mses = []
rmses = []

for name, prediction in [("Naive", naive_forecast),
                         ("Moving Average", ma_forecast),
                         ("Linear Regression", linear_forecast)]:
    mae, mse, rmse = evaluate(name, actual_test, prediction)
    model_names.append(name)
    maes.append(mae)
    mses.append(mse)
    rmses.append(rmse)
print()

plt.figure(figsize=(12, 5))
plt.plot(test.index, actual_test, label="Actual")
plt.plot(test.index, naive_forecast, label="Naive")
plt.plot(test.index, ma_forecast, label="Moving Average")
plt.plot(test.index, linear_forecast, label="Linear Regression")
plt.title("Baseline models")
plt.xlabel("Date")
plt.ylabel("Revenue (EUR)")
plt.legend()
plt.show()

print("LSTM")

train_values = train[["Revenue"]].values
test_values = test[["Revenue"]].values

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_values)
test_scaled = scaler.transform(test_values)


def create_sequences(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


window_size = 14
X_train_lstm, y_train_lstm = create_sequences(train_scaled, window_size)

combined = np.concatenate([train_scaled[-window_size:], test_scaled])
X_test_lstm, y_test_lstm = create_sequences(combined, window_size)

print("X_train_lstm:", X_train_lstm.shape)
print("X_test_lstm: ", X_test_lstm.shape)


model_lstm = Sequential([
    LSTM(50, input_shape=(window_size, 1)),
    Dense(1)
])
model_lstm.compile(optimizer="adam", loss="mse")
model_lstm.summary()

history = model_lstm.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=30,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("LSTM training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.show()

pred_scaled = model_lstm.predict(X_test_lstm)
predictions = scaler.inverse_transform(pred_scaled)
lstm_forecast = predictions[:, 0]

print("LSTM predictions:", len(lstm_forecast), "| Test days:", len(test))

mae, mse, rmse = evaluate("LSTM", actual_test, lstm_forecast)
model_names.append("LSTM")
maes.append(mae)
mses.append(mse)
rmses.append(rmse)
print()

print("Comparison")
results = pd.DataFrame({
    "Model": model_names,
    "MAE": maes,
    "MSE": mses,
    "RMSE": rmses
})
print(results.sort_values("RMSE"))

print("Average daily revenue in the test period:", actual_test.mean())

plt.figure(figsize=(12, 5))
plt.plot(test.index, actual_test, label="Actual", linewidth=2)
plt.plot(test.index, naive_forecast, label="Naive")
plt.plot(test.index, ma_forecast, label="Moving Average")
plt.plot(test.index, linear_forecast, label="Linear Regression")
plt.plot(test.index, lstm_forecast, label="LSTM", linestyle="--")
plt.title("Actual vs forecasts")
plt.xlabel("Date")
plt.ylabel("Revenue (EUR)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

forecast_df = pd.DataFrame({
    "Date": test.index,
    "Actual": actual_test,
    "Naive": naive_forecast,
    "MovingAverage": ma_forecast,
    "LinearRegression": linear_forecast,
    "LSTM": lstm_forecast
})
forecast_df.to_csv("forecast_results.csv", index=False)
results.to_csv("forecast_metrics.csv", index=False)
df.to_csv("daily_series.csv")


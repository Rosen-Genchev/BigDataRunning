import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Running Gear & Supplements - Dashboard", layout="wide")


@st.cache_data
def load_data():
    sales = pd.read_csv("sales_enriched.csv")
    segments = pd.read_csv("customer_segments.csv")
    reviews = pd.read_csv("reviews_sentiment.csv")
    forecast = pd.read_csv("forecast_results.csv")
    metrics = pd.read_csv("forecast_metrics.csv")

    sales["OrderDate"] = pd.to_datetime(sales["OrderDate"])
    reviews["ReviewDate"] = pd.to_datetime(reviews["ReviewDate"])
    forecast["Date"] = pd.to_datetime(forecast["Date"])

    segment_of = {}
    for i in range(len(segments)):
        segment_of[segments["CustomerID"].iloc[i]] = segments["SegmentName"].iloc[i]
    sales["SegmentName"] = [segment_of[c] for c in sales["CustomerID"]]

    return sales, reviews, forecast, metrics


sales, reviews, forecast, metrics = load_data()

st.title("Running Gear & Supplements - Business Dashboard")
st.caption("Revenue counts only completed and shipped orders.")

st.sidebar.header("Filters")

min_date = sales["OrderDate"].min()
max_date = sales["OrderDate"].max()
default_start = max(min_date, max_date - pd.Timedelta(days=91))

date_range = st.sidebar.date_input(
    "Period",
    value=(default_start.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)
if len(date_range) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()

all_categories = sorted(sales["Category"].unique())
selected_categories = st.sidebar.multiselect(
    "Product category", all_categories, default=all_categories)

all_segments = sorted(sales["SegmentName"].unique())
selected_segments = st.sidebar.multiselect(
    "Customer segment", all_segments, default=all_segments)

st.sidebar.caption("The segment filter does not apply to reviews and forecasts.")

start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1])

valid_status = ["completed", "shipped"]

in_filters = (
    sales["Category"].isin(selected_categories)
    & sales["SegmentName"].isin(selected_segments)
)
base = sales[in_filters & sales["Status"].isin(valid_status)]

current = base[(base["OrderDate"] >= start) & (base["OrderDate"] <= end)]
if len(current) == 0:
    st.warning("No sales for the selected filters.")
    st.stop()

period_days = (end - start).days + 1
prev_end = start - pd.Timedelta(days=1)
prev_start = start - pd.Timedelta(days=period_days)
previous = base[(base["OrderDate"] >= prev_start) & (base["OrderDate"] <= prev_end)]

all_current = sales[in_filters & (sales["OrderDate"] >= start) & (sales["OrderDate"] <= end)]
lost_revenue = all_current[~all_current["Status"].isin(valid_status)]["LineTotal"].sum()

revenue = current["LineTotal"].sum()
profit = current["Profit"].sum()
margin = profit / revenue * 100
orders_count = len(current["OrderID"].unique())
customers_count = len(current["CustomerID"].unique())
avg_order_value = revenue / orders_count
units = current["Quantity"].sum()

previous_revenue = previous["LineTotal"].sum()
if prev_start >= min_date and previous_revenue > 0:
    growth = (revenue - previous_revenue) / previous_revenue * 100
    growth_text = f"{growth:+.1f}% vs previous period"
else:
    growth_text = None

row1 = st.columns(4)
row1[0].metric("Total revenue", f"{revenue:,.0f} EUR", delta=growth_text)
row1[1].metric("Total profit", f"{profit:,.0f} EUR")
row1[2].metric("Profit margin", f"{margin:.1f}%")
row1[3].metric("Average order value", f"{avg_order_value:,.0f} EUR")

row2 = st.columns(4)
row2[0].metric("Orders", f"{orders_count:,}")
row2[1].metric("Customers", f"{customers_count:,}")
row2[2].metric("Units sold", f"{units:,.0f}")
row2[3].metric("Lost to cancelled/returned", f"{lost_revenue:,.0f} EUR")

if growth_text is None:
    st.caption("Growth is not shown: there is no earlier period of the same length in the data.")
else:
    st.caption(
        "Growth compares the selected period with the "
        + str(period_days) + " days before it.")

tab_sales, tab_customers, tab_reviews, tab_forecast = st.tabs(
    ["Sales", "Customers", "Reviews", "Forecast"])

with tab_sales:
    monthly = current.set_index("OrderDate")[["LineTotal", "Profit"]].resample("ME").sum()
    monthly_df = pd.DataFrame({
        "Month": monthly.index,
        "Revenue": monthly["LineTotal"].values,
        "Profit": monthly["Profit"].values,
    })
    fig_month = px.line(monthly_df, x="Month", y=["Revenue", "Profit"],
                        markers=True, title="Monthly revenue and profit")
    st.plotly_chart(fig_month)

    left, right = st.columns(2)

    by_category = current.groupby("Category")[["LineTotal", "Profit"]].sum()
    category_df = pd.DataFrame({
        "Category": by_category.index,
        "Revenue": by_category["LineTotal"].values,
        "Profit": by_category["Profit"].values,
    })
    category_df["Margin (%)"] = category_df["Profit"] / category_df["Revenue"] * 100
    fig_cat = px.bar(category_df, x="Category", y=["Revenue", "Profit"],
                     barmode="group", title="Revenue and profit by category")
    left.plotly_chart(fig_cat)

    by_product = current.groupby("ProductName")["Profit"].sum().sort_values(ascending=False).head(10)
    product_df = pd.DataFrame({
        "Product": by_product.index,
        "Profit": by_product.values,
    })
    fig_prod = px.bar(product_df, x="Profit", y="Product", orientation="h",
                      title="Top 10 products by profit")
    fig_prod.update_layout(yaxis={"categoryorder": "total ascending"})
    right.plotly_chart(fig_prod)

    st.subheader("Margin by category")
    st.dataframe(category_df)

with tab_customers:
    left, right = st.columns(2)

    unique_customers = current[["CustomerID", "SegmentName"]].drop_duplicates()
    segment_counts = unique_customers["SegmentName"].value_counts()
    segment_counts_df = pd.DataFrame({
        "Segment": segment_counts.index,
        "Customers": segment_counts.values,
    })
    fig_seg = px.pie(segment_counts_df, names="Segment", values="Customers",
                     title="Customers by segment")
    left.plotly_chart(fig_seg)

    segment_revenue = current.groupby("SegmentName")["LineTotal"].sum()
    segment_revenue_df = pd.DataFrame({
        "Segment": segment_revenue.index,
        "Revenue": segment_revenue.values,
    })
    fig_seg_rev = px.bar(segment_revenue_df, x="Segment", y="Revenue",
                         title="Revenue by segment")
    right.plotly_chart(fig_seg_rev)

with tab_reviews:
    period_reviews = reviews[
        (reviews["ReviewDate"] >= start)
        & (reviews["ReviewDate"] <= end)
        & reviews["Category"].isin(selected_categories)
    ]

    if len(period_reviews) == 0:
        st.info("No reviews for the selected period and categories.")
    else:
        counts = period_reviews["sentiment"].value_counts()
        share = counts / counts.sum() * 100
        sentiment_df = pd.DataFrame({
            "Sentiment": share.index,
            "Share (%)": share.values,
        })

        cols = st.columns(3)
        cols[0].metric("Reviews", len(period_reviews))
        cols[1].metric("Average rating", f"{period_reviews['Rating'].mean():.2f}")
        cols[2].metric("Average compound", f"{period_reviews['compound'].mean():.2f}")

        fig_sent = px.bar(
            sentiment_df, x="Sentiment", y="Share (%)", color="Sentiment",
            color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
            title="Share of sentiment categories")
        st.plotly_chart(fig_sent)
        st.caption("With few reviews in the selected period the shares are not reliable.")

with tab_forecast:
    st.caption("Forecasts are for total daily revenue (all categories and segments), "
               "test period only.")
    chosen_models = st.multiselect(
        "Models to show",
        ["Naive", "MovingAverage", "LinearRegression", "LSTM"],
        default=["MovingAverage", "LSTM"])

    fig_fc = px.line(forecast, x="Date", y=["Actual"] + chosen_models,
                     title="Actual vs forecast daily revenue")
    st.plotly_chart(fig_fc)

    st.subheader("Forecast errors")
    st.dataframe(metrics)
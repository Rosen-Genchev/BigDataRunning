import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)

print("Customer reviews")
print()

print("Loading")
reviews = pd.read_csv("reviews_clean.csv")
products = pd.read_csv("products_clean.csv")

reviews["ReviewDate"] = pd.to_datetime(reviews["ReviewDate"])

print(reviews['ReviewText'].head())
print(reviews.shape)
reviews.info()
print(reviews.isna().sum())
print("Duplicate rows:", reviews.duplicated().sum())

print("\nFirst 10 raw reviews:")
for text in reviews["ReviewText"].head(10):
    print(text)
print()

print("Cleaning")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


reviews["clean_text"] = reviews["ReviewText"].apply(clean_text)
print(reviews[["ReviewText", "clean_text"]].head(15))
print()

print("Tokenization")
stop_words = set(ENGLISH_STOP_WORDS)

for text in reviews["clean_text"].head(10):
    tokens = text.split()
    tokens_clean = [w for w in tokens if w not in stop_words]
    print("Tokens:        ", tokens)
    print("Without stops: ", tokens_clean)
    print()

print("CountVectorizer")
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(reviews["clean_text"])
print("Matrix shape (reviews x words):", X.shape)

word_counts = X.sum(axis=0).A1
word_freq = pd.DataFrame({
    "word": vectorizer.get_feature_names_out(),
    "frequency": word_counts
}).sort_values("frequency", ascending=False)

print("Top-20 words:")
print(word_freq.head(20))

top_words = word_freq.head(15)
plt.figure(figsize=(10, 6))
plt.barh(top_words["word"], top_words["frequency"])
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.title("Most frequent words in reviews")
plt.gca().invert_yaxis()
plt.show()

print("Bigrams")
bigram_vectorizer = CountVectorizer(
    ngram_range=(2, 2),
    stop_words="english"
)
X_bi = bigram_vectorizer.fit_transform(reviews["clean_text"])

bi_counts = X_bi.sum(axis=0).A1
bigrams = pd.DataFrame({
    "ngram": bigram_vectorizer.get_feature_names_out(),
    "frequency": bi_counts
}).sort_values("frequency", ascending=False)

print("Top-15 bigrams:")
print(bigrams.head(15))

top_bigrams = bigrams.head(15)
plt.figure(figsize=(10, 6))
plt.barh(top_bigrams["ngram"], top_bigrams["frequency"])
plt.xlabel("Frequency")
plt.ylabel("Bigram")
plt.title("Most frequent phrases (bigrams) in reviews")
plt.gca().invert_yaxis()
plt.show()

print("TF-IDF")
tfidf = TfidfVectorizer(stop_words="english")
X_tfidf = tfidf.fit_transform(reviews["clean_text"])

tfidf_scores = X_tfidf.sum(axis=0).A1
tfidf_words = pd.DataFrame({
    "word": tfidf.get_feature_names_out(),
    "score": tfidf_scores
}).sort_values("score", ascending=False)

print("Top-20 TF-IDF features:")
print(tfidf_words.head(20))

comparison = pd.DataFrame({
    "CountVectorizer": word_freq["word"].head(15).tolist(),
    "TF-IDF": tfidf_words["word"].head(15).tolist()
})
print("Top-15 compared:")
print(comparison)
print()

print("Sentiment")

sia = SentimentIntensityAnalyzer()

print("Example:", sia.polarity_scores("I love this product!"))


def get_sentiment(text):
    compound = sia.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


reviews["sentiment"] = reviews["clean_text"].apply(get_sentiment)
reviews["compound"] = reviews["clean_text"].apply(
    lambda t: sia.polarity_scores(t)["compound"]
)

print(reviews[["ReviewText", "sentiment", "compound"]].head(10))

sentiment_counts = reviews["sentiment"].value_counts()
print("Counts:")
print(sentiment_counts)

plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.title("Sentiment of reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of reviews")
plt.show()

sentiment_share = reviews["sentiment"].value_counts(normalize=True) * 100
print("Shares (%):")
print(sentiment_share)

plt.bar(sentiment_share.index, sentiment_share.values)
plt.title("Share of sentiment categories")
plt.xlabel("Sentiment")
plt.ylabel("Percent")
plt.show()
print()

print("VADER vs rating")


def rating_to_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"


reviews["rating_sentiment"] = reviews["Rating"].apply(rating_to_sentiment)

print(pd.crosstab(reviews["rating_sentiment"], reviews["sentiment"]))
agreement = (reviews["sentiment"] == reviews["rating_sentiment"]).mean()
print("Agreement between VADER and rating:", agreement)

mismatch = reviews[reviews["sentiment"] != reviews["rating_sentiment"]]
print("Disagreements:", len(mismatch))
print(mismatch[["ReviewText", "Rating", "sentiment", "compound"]].head(10))
print()

print("Sentiment over time")
reviews["Positive"] = (reviews["sentiment"] == "Positive") * 1
reviews["Negative"] = (reviews["sentiment"] == "Negative") * 1
reviews["Neutral"] = (reviews["sentiment"] == "Neutral") * 1

by_date = reviews.set_index("ReviewDate").sort_index()

monthly_counts = by_date[["Positive", "Negative", "Neutral"]].resample("ME").sum()
monthly_total = monthly_counts.sum(axis=1)
print("Monthly counts:")
print(monthly_counts)

monthly_share = pd.DataFrame({
    "Positive": monthly_counts["Positive"] / monthly_total * 100,
    "Negative": monthly_counts["Negative"] / monthly_total * 100,
    "Neutral": monthly_counts["Neutral"] / monthly_total * 100,
})
print("Monthly shares (%):")
print(monthly_share)

monthly_share.plot(figsize=(12, 5))
plt.title("Sentiment shares by month")
plt.xlabel("Month")
plt.ylabel("Percent of reviews")
plt.show()

monthly_compound = by_date["compound"].resample("ME").mean()
print("Average compound by month:")
print(monthly_compound)

plt.figure(figsize=(12, 5))
plt.plot(monthly_compound.index, monthly_compound.values)
plt.axhline(0)
plt.title("Average sentiment over time")
plt.xlabel("Month")
plt.ylabel("Compound sentiment")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print()

print("By category")
category_of = {}
name_of = {}
for i in range(len(products)):
    pid = products["ProductID"].iloc[i]
    category_of[pid] = products["Category"].iloc[i]
    name_of[pid] = products["ProductName"].iloc[i]

reviews["Category"] = [category_of[p] for p in reviews["ProductID"]]
reviews["ProductName"] = [name_of[p] for p in reviews["ProductID"]]

print("Average rating by category:")
print(reviews.groupby("Category")["Rating"].mean())
print("Average compound by category:")
print(reviews.groupby("Category")["compound"].mean())
print(pd.crosstab(reviews["Category"], reviews["sentiment"]))

product_summary = reviews.groupby("ProductName")["Rating"].agg(["mean", "count"])
print("Lowest-rated products:")
print(product_summary.sort_values("mean").head(5))
print("Highest-rated products:")
print(product_summary.sort_values("mean", ascending=False).head(5))
print()

reviews[["ReviewID", "ReviewDate", "ProductID", "ProductName", "Category",
         "Rating", "ReviewText", "clean_text", "sentiment", "compound"]
        ].to_csv("reviews_sentiment.csv", index=False)


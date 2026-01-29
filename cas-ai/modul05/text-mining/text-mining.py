import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

import nltk
nltk.download("vader_lexicon")

###
# Getting started:
# pip install pandas nltk matplotlib
###

# -----------------------------
# 1. CSV laden
# -----------------------------
df = pd.read_csv("donal_trump_tweet_history_2009_2025/djt_posts_dec2025.csv")

# Datum parsen
df["date"] = pd.to_datetime(df["date"], utc=True)
df["year"] = df["date"].dt.year

# Nur Originalposts (keine Reposts)
df = df[df["repost_flag"] == False]

# -----------------------------
# 2. Text Cleaning (minimal, erklärbar)
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)   # URLs entfernen
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# -----------------------------
# 3. Feature Engineering
# -----------------------------

# 3.1 Sentiment (VADER)
sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["clean_text"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)

# 3.2 ALL CAPS Anteil
def caps_ratio(text):
    if not isinstance(text, str):
        return 0.0

    words = re.findall(r"\b[A-Z]{2,}\b", text)
    total_words = len(text.split())

    return len(words) / total_words if total_words > 0 else 0.0


df["caps_ratio"] = df["text"].apply(caps_ratio)

# 3.3 Ausrufezeichen
df["exclamation_count"] = df["text"].str.count("!")

# 3.4 Superlative / extreme Begriffe (einfaches Lexikon)
SUPERLATIVES = [
    "best", "worst", "greatest", "tremendous",
    "disaster", "total", "incredible", "fraud"
]

def contains_superlative(text):
    return any(word in text for word in SUPERLATIVES)

df["has_superlative"] = df["clean_text"].apply(contains_superlative)

# -----------------------------
# 4. Aggregation pro Jahr
# -----------------------------
yearly = df.groupby("year").agg({
    "sentiment": "mean",
    "caps_ratio": "mean",
    "exclamation_count": "mean",
    "has_superlative": "mean",
    "text": "count"
}).rename(columns={"text": "post_count"}).reset_index()

# -----------------------------
# 5. Visualisierung
# -----------------------------

plt.figure()
plt.plot(yearly["year"], yearly["sentiment"])
plt.title("Average Sentiment per Year")
plt.xlabel("Year")
plt.ylabel("Sentiment (VADER)")
plt.savefig('m05_01_txt_mining_sentiment.png', dpi=300)
plt.show()

plt.figure()
plt.plot(yearly["year"], yearly["caps_ratio"])
plt.title("Average ALL CAPS Ratio per Year")
plt.xlabel("Year")
plt.ylabel("CAPS Ratio")
plt.savefig('m05_01_caps_ratio.png', dpi=300)
plt.show()

plt.figure()
plt.plot(yearly["year"], yearly["exclamation_count"])
plt.title("Average ALL CAPS Ratio per Year")
plt.xlabel("Year")
plt.ylabel("CAPS Ratio")
plt.savefig('m05_01_exclamation_count.png', dpi=300)
plt.show()

plt.figure()
plt.plot(yearly["year"], yearly["has_superlative"])
plt.title("Share of Posts with Superlatives")
plt.xlabel("Year")
plt.ylabel("Proportion")
plt.savefig('m05_01_superlatives_proportion.png', dpi=300)
plt.show()

# -----------------------------
# 6. Ausgabe für Interpretation
# -----------------------------
print(yearly)

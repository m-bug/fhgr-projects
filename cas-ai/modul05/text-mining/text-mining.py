import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

###
# Getting started:
# pip install pandas nltk matplotlib wordcloud
###

# =====================================================
# 1. Daten laden & vorbereiten
# =====================================================
df = pd.read_csv("donald_trump_tweet_history_2009_2025/djt_posts_dec2025.csv")

# Datum parsen
df["date"] = pd.to_datetime(df["date"], utc=True)
df["year"] = df["date"].dt.year

# Nur Originalposts
df = df[df["repost_flag"] == False]

# Leere Texte abfangen
df["text"] = df["text"].fillna("")
df["word_count"] = df["word_count"].fillna(0)

# nur echte Textposts
df = df[df["word_count"] > 0]

# =====================================================
# 2. Text Cleaning (minimal & transparent)
# =====================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# =====================================================
# 3. Sentiment (VADER)
# =====================================================

sia = SentimentIntensityAnalyzer()

df["sentiment"] = df["clean_text"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)

# =====================================================
# 4. Emphase-Features
# =====================================================

# 4.1 ALL CAPS Ratio
def caps_ratio(text):
    if not isinstance(text, str):
        return 0.0
    caps_words = re.findall(r"\b[A-Z]{2,}\b", text)
    total_words = len(text.split())
    return len(caps_words) / total_words if total_words > 0 else 0.0

df["caps_ratio"] = df["text"].apply(caps_ratio)

# 4.2 Ausrufezeichen
df["exclamation_count"] = df["text"].str.count("!")

# =====================================================
# 5. Extreme Language – saubere Definition
# =====================================================

SUPERLATIVES = [
    "best", "worst", "greatest", "biggest", "strongest",
    "weakest", "largest", "smallest", "highest", "lowest"
]

INTENSIFIERS = [
    "very", "totally", "extremely", "absolutely",
    "completely", "highly", "incredibly", "tremendously"
]

ABSOLUTES = [
    "never", "always", "everyone", "nobody",
    "nothing", "everything", "all", "none"
]

def count_matches(text, lexicon):
    if not isinstance(text, str):
        return 0
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    return sum(1 for t in tokens if t in lexicon)

df["superlative_count"] = df["clean_text"].apply(
    lambda x: count_matches(x, SUPERLATIVES)
)

df["intensifier_count"] = df["clean_text"].apply(
    lambda x: count_matches(x, INTENSIFIERS)
)

df["absolute_count"] = df["clean_text"].apply(
    lambda x: count_matches(x, ABSOLUTES)
)

# Normalisierung pro 100 Wörter
df["superlatives_per_100w"] = (df["superlative_count"] / df["word_count"]) * 100
df["intensifiers_per_100w"] = (df["intensifier_count"] / df["word_count"]) * 100
df["absolutes_per_100w"] = (df["absolute_count"] / df["word_count"]) * 100

# =====================================================
# 6. Aggregation pro Jahr
# =====================================================

yearly = df.groupby("year").agg({
    "sentiment": ["mean", "std"],
    "caps_ratio": "mean",
    "exclamation_count": "mean",
    "superlatives_per_100w": "mean",
    "intensifiers_per_100w": "mean",
    "absolutes_per_100w": "mean",
    "text": "count"
}).reset_index()

yearly.columns = [
    "year",
    "sentiment_mean",
    "sentiment_std",
    "caps_ratio",
    "exclamation_count",
    "superlatives_per_100w",
    "intensifiers_per_100w",
    "absolutes_per_100w",
    "post_count"
]

print(yearly.tail())

# =====================================================
# 7. Visualisierungen
# =====================================================

# Sentiment Mittelwert
plt.figure()
plt.plot(yearly["year"], yearly["sentiment_mean"])
plt.title("Average Sentiment per Year")
plt.xlabel("Year")
plt.ylabel("Sentiment (VADER)")
plt.savefig('m05_01_mean_sentiment.png', dpi=300)
plt.show()

# Sentiment Varianz
plt.figure()
plt.plot(yearly["year"], yearly["sentiment_std"])
plt.title("Sentiment Variability per Year")
plt.xlabel("Year")
plt.ylabel("Sentiment Standard Deviation")
plt.savefig('m05_01_std_sentiment.png', dpi=300)
plt.show()

# ALL CAPS
plt.figure()
plt.plot(yearly["year"], yearly["caps_ratio"])
plt.title("Average ALL CAPS Ratio per Year (G1)")
plt.xlabel("Year")
plt.ylabel("CAPS Ratio")
plt.savefig('m05_01_caps_ratio.png', dpi=300)
plt.show()

# Exclamation
plt.figure()
plt.plot(yearly["year"], yearly["exclamation_count"])
plt.title("Average Exclamation Count per Year")
plt.xlabel("Year")
plt.ylabel("Exclamation Count")
plt.savefig('m05_01_exclamation_count.png', dpi=300)
plt.show()

# Extreme Language
plt.figure()
plt.plot(yearly["year"], yearly["superlatives_per_100w"], label="Superlatives")
plt.plot(yearly["year"], yearly["intensifiers_per_100w"], label="Intensifiers")
plt.plot(yearly["year"], yearly["absolutes_per_100w"], label="Absolutes")
plt.title("Extreme Language per 100 Words (G2)")
plt.xlabel("Year")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('m05_01_extreme_lang.png', dpi=300)
plt.show()

# =====================================================
# 8. Wortfrequenzen (gesamt)
# =====================================================

from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import nltk

# offical nltk stopwords, besser als manuelles abfüllen
nltk.download("stopwords")

# ---------------------------------------------
# 8.1 Stopwords definieren
# ---------------------------------------------

STOPWORDS = set(stopwords.words("english"))

# Korpus-spezifische Stopwords, twitter/truthsocial overhead, und handles müssen auch weg
CUSTOM_STOPWORDS = {
    "trump", "donald", "realdonaldtrump",
    "twitter", "pic", "image", "rt",
    "amp", "https", "http", "com", "get"
}

ALL_STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)

# ---------------------------------------------
# 8.2 Tokenisierung & Frequenzen
# ---------------------------------------------

tokens = re.findall(r"\b[a-z]+\b", " ".join(df["clean_text"]))

# laenge >2 um rauschen zu vermeiden
filtered_tokens = [
    t for t in tokens
    if t not in ALL_STOPWORDS and len(t) > 2
]

word_freq = Counter(filtered_tokens)

# Top 20
top_words = pd.DataFrame(
    word_freq.most_common(20),
    columns=["word", "count"]
)

# debug top 20
print(top_words)


# =====================================================
# 9. Visualisierung – häufigste Wörter
# =====================================================

# ---------------------------------------------
# 9.1 Wordcloud – gesamter Zeitraum
# ---------------------------------------------

wc = WordCloud(
    width=1400,
    height=700,
    background_color="white",
    stopwords=ALL_STOPWORDS,
    max_words=150,
    min_font_size=10,
    collocations=False  # default ist true, wordcloud probiert dann bigramms zu erkennen, möchten wir hier nicht 
)

wc.generate_from_frequencies(word_freq)

plt.figure(figsize=(14, 7))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Wordcloud – Most Frequent Words (Trump Tweets 2009–2025)")
plt.savefig("m05_01_wordcloud_all.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------
# 9.2 Bigramme bilden
# ---------------------------------------------

bigrams = zip(filtered_tokens, filtered_tokens[1:])
bigram_freq = Counter(
    [" ".join(b) for b in bigrams]
)

wc_bigram = WordCloud(
    width=1400,
    height=700,
    background_color="white",
    max_words=100,
    collocations=False # auch hier: bigrams kommen von mir und wordcloud soll hier nichts selber machen
)

wc_bigram.generate_from_frequencies(bigram_freq)

plt.figure(figsize=(14, 7))
plt.imshow(wc_bigram, interpolation="bilinear")
plt.axis("off")
#plt.title("Bigram Wordcloud – Frequent Phrases")
plt.savefig("m05_01_wordcloud_bigrams.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------
# 9.3.1 Top-Unigramme (Balkendiagramm)
# ---------------------------------------------

top_unigrams = word_freq.most_common(20)
unigram_df = pd.DataFrame(top_unigrams, columns=["token", "count"])

plt.figure(figsize=(10, 6))
plt.barh(unigram_df["token"], unigram_df["count"])
plt.gca().invert_yaxis()
plt.title("Top 20 Most Frequent Unigrams (2009–2025)")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig("m05_02_top_unigrams_bar.png", dpi=300)
plt.show()

# ---------------------------------------------
# 9.3.2 Top-Bigramme (Balkendiagramm)
# ---------------------------------------------

top_bigrams = bigram_freq.most_common(20)
bigram_df = pd.DataFrame(top_bigrams, columns=["token", "count"])

plt.figure(figsize=(10, 6))
plt.barh(bigram_df["token"], bigram_df["count"])
plt.gca().invert_yaxis()
plt.title("Top 20 Most Frequent Bigrams (2009–2025)")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig("m05_02_top_bigrams_bar.png", dpi=300)
plt.show()

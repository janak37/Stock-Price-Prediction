import pandas as pd
from sentiment.naive_bayes import NaiveBayes

def process_sentiment(symbol):
    news_path = f"data/raw/news/{symbol}_news.csv"
    news = pd.read_csv(news_path, encoding="ISO-8859-1")
    training = pd.read_csv("training_sentiment.csv")
    
    nb = NaiveBayes()
    nb.fit(training["Text"].values, training["Sentiment"].values)

    news["Label"] = news["Title"].apply(lambda x: nb.predict(x))

    mapping = {
        "Strong_Bullish": 1,
        "Neutral": 0,
        "Weak_Bullish": -1
    }

    news["Sentiment"] = news["Label"].map(mapping)
    daily_sentiment = news.groupby("Date")["Sentiment"].mean().reset_index()

    output_path = f"data/processed/sentiment/{symbol}_sentiment.csv"
    daily_sentiment.to_csv(output_path, index=False)
    print(f"(symbol) sentiment generated successfully.")
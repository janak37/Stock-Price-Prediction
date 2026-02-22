import os
from sentiment.sentiment_processor import process_sentiment

news_folder = "data/raw/news"
for file in os.listdir(news_folder):
    if file.endswith("_news.csv"):
        symbol = file.replace("_news.csv", "")
        process_sentiment(symbol)

print("All sentiment files generated.")
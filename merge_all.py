import os
import pandas as pd

stock_folder = "data/raw/stock"
sentiment_folder = "data/processed/sentiment"
output_folder = "data/processed/merged"

for file in os.listdir(stock_folder):

    if file.endswith("_stock.csv"):

        symbol = file.replace("_stock.csv", "")

        stock_path = f"{stock_folder}/{symbol}_stock.csv"
        sentiment_path = f"{sentiment_folder}/{symbol}_sentiment.csv"

        if os.path.exists(sentiment_path):

            stock = pd.read_csv(stock_path, thousands=",")
            sentiment = pd.read_csv(sentiment_path)

            
            stock["Date"] = pd.to_datetime(stock["Date"])
            sentiment["Date"] = pd.to_datetime(sentiment["Date"])

            
            merged = pd.merge(stock, sentiment, on="Date", how="inner")

            
            merged = merged.sort_values("Date")

            
            output_path = f"{output_folder}/{symbol}_final.csv"
            merged.to_csv(output_path, index=False)

            print(f"{symbol} merged successfully.")

        else:
            print(f"Sentiment file missing for {symbol}")

print("All merging completed.")
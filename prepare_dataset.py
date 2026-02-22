import os
import pandas as pd

merged_folder = "data/processed/merged"
output_folder = "data/processed/merged"

for file in os.listdir(merged_folder):

    if file.endswith("_final.csv"):

        symbol = file.replace("_final.csv", "")
        path = f"{merged_folder}/{file}"

        
        df = pd.read_csv(path, thousands=",")

        
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        
        numeric_cols = ["Open", "High", "Low", "Qty", "Ltp", "Sentiment"]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        
        df["Target"] = df["Ltp"].shift(-1)

        
        df = df.dropna()

        
        df = df[["Date", "Open", "High", "Low", "Qty", "Sentiment", "Target"]]

        
        df.to_csv(f"{output_folder}/{symbol}_train.csv", index=False)

        print(f"{symbol} dataset prepared.")

print("All datasets prepared.")
import pandas as pd
import os

folder = "data/raw/stock"

files = os.listdir(folder)

for file in files:

    if file.endswith(".csv"):

        path = os.path.join(folder, file)

        df = pd.read_csv(path)

        print("Cleaning:", file)

       
        df = df.drop(columns=["S.N."], errors="ignore")

       
        df = df.drop(columns=["% Change", "Turnover"], errors="ignore")

        df.to_csv(path, index=False)

print("All stock files cleaned successfully.")
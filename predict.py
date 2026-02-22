import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

merged_path = os.path.join(project_root, "data", "processed", "merged")
model_dir = os.path.join(project_root, "model")

FEATURES = ["open", "high", "low", "qty", "sentiment"]


available = [f.replace("_model.txt", "") for f in os.listdir(model_dir) if f.endswith("_model.txt")]
print("Available companies:")
for i, name in enumerate(available):
    print(f"  {i+1}. {name}")


company = input("\nEnter company symbol to predict: ").strip().upper()

if company not in available:
    print(f"No model found for {company}")
    exit()


params = {}
with open(os.path.join(model_dir, f"{company}_model.txt"), "r") as f:
    for line in f:
        key, val = line.strip().split("=")
        params[key] = float(val)

intercept = params["intercept"]
weights = [params[feat] for feat in FEATURES]


data = pd.read_csv(os.path.join(merged_path, f"{company}_train.csv"))
data.columns = data.columns.str.strip().str.lower()
latest = data.tail(1).iloc[0]

print(f"\nLatest data for {company}:")
for f in FEATURES:
    print(f"  {f}: {latest[f]}")


predicted = intercept + sum(weights[i] * float(latest[FEATURES[i]]) for i in range(len(FEATURES)))
print(f"\nPredicted Next LTP for {company}: {predicted:.2f}")
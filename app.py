import os 
import math 
import pandas as pd
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

Base_DIR = os.path.dirname(os.path.abspath(__file__))
merged_path = os.path.join(Base_DIR, "data", "processed", "merged")
linear_model_dir = os.path.join(Base_DIR, "model", "Linear")
ann_model_dir = os.path.join(Base_DIR, "model", "ann")

features = ["open", "high", "low", "qty", "sentiment"]
hidden_size = 8
input_size = len(features)

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1/(1 + math.exp(-x))

def scale(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return(value-min-val)/(max_val-min_val)

def unscale(value, min_val, max_val):
    return value * (max_val-min_val)+min_val

def get_companies():
    companies = []
    if os.path.exists(ann_model_dir):
        for f in os.listdir(ann_model_dir):
            if f.endswith("_ann.txt"):
                companies.append(f.replace("_ann.txt", ""))
    return sorted(companies)

def predict_linear(company, latest):
    model_path = os.path.join(linear_model_dir, f"{company}_model.txt")

    if not os.path.exists(model_path):
        return None

    params = {}
    with open(model_path, "r") as f:
        for line in f:
            key, val = line.strip().split("=")
            params[key] = float(val)

    intercept = params["intercept"]
    weights = [params[feat] for feat in features]

    predicted = intercept + sum(weights[i] * float(latest[features[i]]) for i in range(len(features)))
    return round(predicted, 2)   

def predict_ann(company, latest):
    model_path = os.path.join(ann_model_dir, f"{company}_ann.txt")

    if not os.path.exists(model_path):
        return None
    params = {}
    with open(model_path, "r") as f:
        for line in f:
            key, val = line.strip().split("=")
            params[key] = float(val)

    
    w1 = [[params[f"w1_{j}_{h}"] for h in range(hidden_size)] for j in range(input_size)]
    b1 = [params[f"b1_{h}"] for h in range(hidden_size)]
    w2 = [params[f"w2_{h}"] for h in range(hidden_size)]
    b2 = params["b2"]

    target_min = params["target_min"]
    target_max = params["target_max"]

    scaled_input = [scale(float(latest[f]), params[f"{f}_min"], params[f"{f}_max"]) for f in features]

    
    hidden = []
    for h in range(hidden_size):
        val = b1[h]
        for j in range(input_size):
            val += scaled_input[j] * w1[j][h]
        hidden.append(sigmoid(val)) 

    output = b2
    for h in range(hidden_size):
        output += hidden[h] * w2[h]
    output = sigmoid(output)

    
    predicted = unscale(output, target_min, target_max)
    return round(predicted, 2)      

def get_chart_data(company):
    train_path = os.path.join(merged_path, f"{company}_train.csv")

    if not os.path.exists(train_path):
        return None

    data = pd.read_csv(train_path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.sort_values("date")

    data["ma20"] = data["ltp"].rolling(window=20).mean().round(2)

    
    delta = data["ltp"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data["rsi"] = (100 - (100 / (1 + rs))).round(2)

    data = data.fillna(0)    

    chart_data = {
        "dates": data["date"].astype(str).tolist(),
        "ltp": data["ltp"].tolist(),
        "open": data["open"].tolist(),
        "high": data["high"].tolist(),
        "low": data["low"].tolist(),
        "qty": data["qty"].tolist(),
        "sentiment": data["sentiment"].tolist(),
        "ma20": data["ma20"].tolist(),
        "rsi": data["rsi"].tolist(),
    }

    return chart_data  
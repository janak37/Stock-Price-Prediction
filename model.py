import os
import pandas as pd
import math
import random



current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

merged_path = os.path.join(project_root, "data", "processed", "merged")
ann_model_dir = os.path.join(project_root, "model", "ann")
os.makedirs(ann_model_dir, exist_ok=True)

print("Reading training data from:", merged_path)
print()

if not os.path.exists(merged_path):
    print("ERROR: Merged folder not found.")
    exit()

train_files = [f for f in os.listdir(merged_path) if f.endswith("_train.csv")]

if len(train_files) == 0:
    print("No training files found.")
    exit()



def sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)



def get_stats(data, cols):
    stats = {}
    for c in cols:
        vals = list(data[c])
        stats[c] = {"min": min(vals), "max": max(vals)}
    return stats

def scale(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def unscale(value, min_val, max_val):
    return value * (max_val - min_val) + min_val



FEATURES = ["open", "high", "low", "qty", "sentiment"]
INPUT_SIZE = len(FEATURES)   
HIDDEN_SIZE = 8              
OUTPUT_SIZE = 1              
LEARNING_RATE = 0.01
EPOCHS = 1000



for file in train_files:

    company = file.replace("_train.csv", "")
    print(f"Training ANN: {company}")

    data = pd.read_csv(os.path.join(merged_path, file))
    data.columns = data.columns.str.strip().str.lower()

    missing = [f for f in FEATURES + ["target"] if f not in data.columns]
    if missing:
        print(f"  Skipping - Missing columns: {missing}\n")
        continue

    data = data.dropna(subset=FEATURES + ["target"])
    n = len(data)

    if n == 0:
        print(f"  Skipping - No usable data\n")
        continue

    
    stats = get_stats(data, FEATURES)
    target_min = float(data["target"].min())
    target_max = float(data["target"].max())

    
    X = []
    y = []
    for i in range(n):
        row = [scale(float(data.iloc[i][f]), stats[f]["min"], stats[f]["max"]) for f in FEATURES]
        X.append(row)
        y.append(scale(float(data.iloc[i]["target"]), target_min, target_max))

    
    random.seed(42)

    
    w1 = [[random.uniform(-0.5, 0.5) for _ in range(HIDDEN_SIZE)] for _ in range(INPUT_SIZE)]
    b1 = [random.uniform(-0.5, 0.5) for _ in range(HIDDEN_SIZE)]

    
    w2 = [random.uniform(-0.5, 0.5) for _ in range(HIDDEN_SIZE)]
    b2 = random.uniform(-0.5, 0.5)

  
    for epoch in range(EPOCHS):
        total_loss = 0

        for i in range(n):
           
            hidden = []
            for h in range(HIDDEN_SIZE):
                val = b1[h]
                for j in range(INPUT_SIZE):
                    val += X[i][j] * w1[j][h]
                hidden.append(sigmoid(val))

         
            output = b2
            for h in range(HIDDEN_SIZE):
                output += hidden[h] * w2[h]
            output = sigmoid(output)

            
            error = y[i] - output
            total_loss += error ** 2

            
            d_output = error * sigmoid_derivative(output)

            
            d_hidden = []
            for h in range(HIDDEN_SIZE):
                d_hidden.append(d_output * w2[h] * sigmoid_derivative(hidden[h]))

            
            for h in range(HIDDEN_SIZE):
                w2[h] += LEARNING_RATE * d_output * hidden[h]
            b2 += LEARNING_RATE * d_output

            
            for h in range(HIDDEN_SIZE):
                for j in range(INPUT_SIZE):
                    w1[j][h] += LEARNING_RATE * d_hidden[h] * X[i][j]
                b1[h] += LEARNING_RATE * d_hidden[h]

        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / n
            print(f"  Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.6f}")

   
    predictions = []
    for i in range(n):
        hidden = []
        for h in range(HIDDEN_SIZE):
            val = b1[h]
            for j in range(INPUT_SIZE):
                val += X[i][j] * w1[j][h]
            hidden.append(sigmoid(val))
        output = b2
        for h in range(HIDDEN_SIZE):
            output += hidden[h] * w2[h]
        predictions.append(sigmoid(output))

    mean_y = sum(y) / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum((y[i] - predictions[i]) ** 2 for i in range(n))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    print(f"  R² Score: {r2:.4f}")

   
    model_path = os.path.join(ann_model_dir, f"{company}_ann.txt")
    with open(model_path, "w") as f:
        
        for feat in FEATURES:
            f.write(f"{feat}_min={stats[feat]['min']}\n")
            f.write(f"{feat}_max={stats[feat]['max']}\n")
        f.write(f"target_min={target_min}\n")
        f.write(f"target_max={target_max}\n")

        
        for j in range(INPUT_SIZE):
            for h in range(HIDDEN_SIZE):
                f.write(f"w1_{j}_{h}={w1[j][h]}\n")

        
        for h in range(HIDDEN_SIZE):
            f.write(f"b1_{h}={b1[h]}\n")

        
        for h in range(HIDDEN_SIZE):
            f.write(f"w2_{h}={w2[h]}\n")

        
        f.write(f"b2={b2}\n")

    print(f"  Model saved → model/ann/{company}_ann.txt\n")

print("All companies ANN training completed.")
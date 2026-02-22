import pandas as pd
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

merged_path = os.path.join(project_root, "data", "processed", "merged")
model_dir = os.path.join(project_root, "model")
os.makedirs(model_dir, exist_ok=True)

print("Reading training data from:", merged_path)
print()

if not os.path.exists(merged_path):
    print("ERROR: Merged folder not found.")
    exit()

files = os.listdir(merged_path)
train_files = [f for f in files if f.endswith("_train.csv")]

if len(train_files) == 0:
    print("No training files found.")
    exit()

print("Training started for all companies...\n")



def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def mat_mul(A, B):
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    result = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def mat_inverse(matrix):
    n = len(matrix)
    aug = [matrix[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = -1
        for row in range(col, n):
            if abs(aug[row][col]) > 1e-10:
                pivot = row
                break
        if pivot == -1:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [x / scale for x in aug[col]]
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(2 * n)]
    return [[aug[i][n + j] for j in range(n)] for i in range(n)]



FEATURES = ["open", "high", "low", "qty", "sentiment"]

for file in train_files:

    company = file.replace("_train.csv", "")
    print(f"Training: {company}")

    file_path = os.path.join(merged_path, file)
    data = pd.read_csv(file_path)
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

    
    X = [[1.0] + [float(data.iloc[i][f]) for f in FEATURES] for i in range(n)]
    y = [[float(data.iloc[i]["target"])] for i in range(n)]

   
    XT = transpose(X)
    XTX = mat_mul(XT, X)
    XTX_inv = mat_inverse(XTX)

    if XTX_inv is None:
        print(f"  Skipping - Matrix is singular\n")
        continue

    XTy = mat_mul(XT, y)
    coefficients = mat_mul(XTX_inv, XTy)

    intercept = coefficients[0][0]
    weights = [coefficients[i + 1][0] for i in range(len(FEATURES))]

    
    y_pred = [(intercept + sum(weights[j] * X[i][j+1] for j in range(len(FEATURES)))) for i in range(n)]
    y_actual = [y[i][0] for i in range(n)]
    mean_y = sum(y_actual) / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y_actual)
    ss_res = sum((y_actual[i] - y_pred[i]) ** 2 for i in range(n))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    print(f"  Intercept : {intercept:.4f}")
    for i, f in enumerate(FEATURES):
        print(f"  {f} weight : {weights[i]:.4f}")
    print(f"  R² Score  : {r2:.4f}")

    
    model_file_path = os.path.join(model_dir, f"{company}_model.txt")
    with open(model_file_path, "w") as f:
        f.write(f"intercept={intercept}\n")
        for i, feat in enumerate(FEATURES):
            f.write(f"{feat}={weights[i]}\n")

    print(f"  Model saved → model/{company}_model.txt\n")

print("All companies training completed.")
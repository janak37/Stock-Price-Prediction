import os
import math
import pandas as pd
from flask import Flask, send_from_directory, request, redirect, url_for, session, jsonify
from companies.company_list import CATEGORIZED_COMPANIES

app = Flask(__name__)
app.secret_key = "sharesathi_secret_2024"

# ---------------------------------------------------
# PATHS & CONSTANTS
# ---------------------------------------------------
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
merged_path      = os.path.join(BASE_DIR, "data", "processed", "merged")
linear_model_dir = os.path.join(BASE_DIR, "model")
ann_model_dir    = os.path.join(BASE_DIR, "model", "ann")
training_csv     = os.path.join(BASE_DIR, "training_sentiment.csv")

FEATURES    = ["open", "high", "low", "qty", "sentiment"]
HIDDEN_SIZE = 8
INPUT_SIZE  = len(FEATURES)

ADMIN_USER = "admin"
ADMIN_PASS = "admin123"

# ---------------------------------------------------
# HELPER: Read HTML file from project root
# ---------------------------------------------------
def read_html(filename):
    """Reads an HTML file from the same folder as app.py"""
    path = os.path.join(BASE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def serve_page(filename):
    """Serve a static HTML page from the project root."""
    return send_from_directory(BASE_DIR, filename)

# ---------------------------------------------------
# MATH HELPERS
# ---------------------------------------------------
def sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

def scale(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def unscale(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

# ---------------------------------------------------
# COLUMN NORMALISER
# ---------------------------------------------------
def fix_columns(data):
    data.columns = [str(c).strip().lower() for c in data.columns]
    rename_map = {}
    for col in data.columns:
        c = col.strip().lower()
        if c in ["close", "closing price", "close price", "ltp", "last traded price", "last price"]:
            rename_map[col] = "ltp"
        elif c in ["volume", "qty", "quantity", "shares traded", "traded quantity"]:
            rename_map[col] = "qty"
        elif c in ["open", "open price", "opening price"]:
            rename_map[col] = "open"
        elif c in ["high", "high price", "day high"]:
            rename_map[col] = "high"
        elif c in ["low", "low price", "day low"]:
            rename_map[col] = "low"
        elif c in ["date", "trade date", "trading date"]:
            rename_map[col] = "date"
        elif c == "sentiment":
            rename_map[col] = "sentiment"
    data = data.rename(columns=rename_map)
    if "ltp" not in data.columns and "close" in data.columns:
        data["ltp"] = data["close"]
    if "qty" not in data.columns and "volume" in data.columns:
        data["qty"] = data["volume"]
    if "ltp" not in data.columns and "target" in data.columns:
        data["ltp"] = data["target"]
    return data

# ---------------------------------------------------
# DATA FUNCTIONS
# ---------------------------------------------------
def get_companies():
    companies = []
    if os.path.exists(ann_model_dir):
        for f in os.listdir(ann_model_dir):
            if f.endswith("_ann.txt"):
                companies.append(f.replace("_ann.txt", ""))
    return sorted(companies)

def get_sentiment_data():
    sentiment_data = {}
    if os.path.exists(merged_path):
        for f in os.listdir(merged_path):
            if f.endswith("_train.csv"):
                symbol = f.replace("_train.csv", "")
                try:
                    df = pd.read_csv(os.path.join(merged_path, f))
                    df.columns = df.columns.str.strip().str.lower()
                    if "sentiment" in df.columns:
                        score = round(float(df["sentiment"].iloc[-1]), 2)
                        sentiment_data[symbol] = score
                except Exception:
                    pass
    return sentiment_data

def sentiment_signal(score):
    if score >= 0.75:
        return "STRONG BULLISH", "sbull"
    elif score >= 0.25:
        return "WEAK BULLISH", "wbull"
    elif score > -0.25:
        return "NEUTRAL", "neutral"
    elif score > -0.75:
        return "WEAK BEARISH", "wbear"
    else:
        return "STRONG BEARISH", "sbear"

def predict_linear(company, latest):
    model_path = os.path.join(linear_model_dir, f"{company}_model.txt")
    if not os.path.exists(model_path):
        return None, None
    params = {}
    with open(model_path) as f:
        for line in f:
            k, v = line.strip().split("=")
            params[k] = float(v)
    intercept = params["intercept"]
    weights   = [params[feat] for feat in FEATURES]
    r2        = params.get("r2", 0)
    predicted = intercept + sum(weights[i] * float(latest[FEATURES[i]]) for i in range(len(FEATURES)))
    return round(predicted, 2), round(r2 * 100, 2)

def predict_ann(company, latest):
    model_path = os.path.join(ann_model_dir, f"{company}_ann.txt")
    if not os.path.exists(model_path):
        return None, None
    params = {}
    with open(model_path) as f:
        for line in f:
            k, v = line.strip().split("=")
            params[k] = float(v)
    w1 = [[params[f"w1_{j}_{h}"] for h in range(HIDDEN_SIZE)] for j in range(INPUT_SIZE)]
    b1 = [params[f"b1_{h}"] for h in range(HIDDEN_SIZE)]
    w2 = [params[f"w2_{h}"] for h in range(HIDDEN_SIZE)]
    b2 = params["b2"]
    r2 = params.get("r2", 0)
    target_min = params["target_min"]
    target_max = params["target_max"]
    scaled_input = [scale(float(latest[f]), params[f"{f}_min"], params[f"{f}_max"]) for f in FEATURES]
    hidden = []
    for h in range(HIDDEN_SIZE):
        val = b1[h] + sum(scaled_input[j] * w1[j][h] for j in range(INPUT_SIZE))
        hidden.append(sigmoid(val))
    output = sigmoid(b2 + sum(hidden[h] * w2[h] for h in range(HIDDEN_SIZE)))
    return round(unscale(output, target_min, target_max), 2), round(r2 * 100, 2)

def get_chart_data(company):
    train_path = os.path.join(merged_path, f"{company}_train.csv")
    if not os.path.exists(train_path):
        return None
    data = pd.read_csv(train_path)
    data = fix_columns(data)
    data = data.sort_values("date").reset_index(drop=True)
    missing = [c for c in ["ltp", "open", "high", "low", "qty", "sentiment"] if c not in data.columns]
    if missing:
        return None
    data["ma20"] = data["ltp"].rolling(window=20).mean().round(2)
    delta = data["ltp"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = -delta.where(delta < 0, 0).rolling(14).mean()
    data["rsi"] = (100 - (100 / (1 + gain / loss))).round(2)
    latest_score = float(data["sentiment"].iloc[-1])
    signal, sig_class = sentiment_signal(latest_score)
    data = data.fillna(0)
    return {
        "dates":            data["date"].astype(str).tolist(),
        "ltp":              data["ltp"].tolist(),
        "open":             data["open"].tolist(),
        "high":             data["high"].tolist(),
        "low":              data["low"].tolist(),
        "qty":              data["qty"].tolist(),
        "sentiment":        data["sentiment"].tolist(),
        "ma20":             data["ma20"].tolist(),
        "rsi":              data["rsi"].tolist(),
        "signal":           signal,
        "sig_class":        sig_class,
        "latest_sentiment": latest_score,
    }

def get_comparison_data(company):
    train_path = os.path.join(merged_path, f"{company}_train.csv")
    if not os.path.exists(train_path):
        return None, None
    data = pd.read_csv(train_path)
    data = fix_columns(data)
    data = data.sort_values("date").reset_index(drop=True)
    n = len(data)
    lr_preds, ann_preds = [], []

    lp = os.path.join(linear_model_dir, f"{company}_model.txt")
    if os.path.exists(lp):
        p = {}
        with open(lp) as f:
            for line in f:
                k, v = line.strip().split("="); p[k] = float(v)
        intercept = p["intercept"]
        weights   = [p[feat] for feat in FEATURES]
        for i in range(n):
            lr_preds.append(round(intercept + sum(weights[j] * float(data.iloc[i][FEATURES[j]]) for j in range(len(FEATURES))), 2))

    ap = os.path.join(ann_model_dir, f"{company}_ann.txt")
    if os.path.exists(ap):
        p = {}
        with open(ap) as f:
            for line in f:
                k, v = line.strip().split("="); p[k] = float(v)
        w1 = [[p[f"w1_{j}_{h}"] for h in range(HIDDEN_SIZE)] for j in range(INPUT_SIZE)]
        b1 = [p[f"b1_{h}"] for h in range(HIDDEN_SIZE)]
        w2 = [p[f"w2_{h}"] for h in range(HIDDEN_SIZE)]
        b2 = p["b2"]
        tmin, tmax = p["target_min"], p["target_max"]
        for i in range(n):
            si  = [scale(float(data.iloc[i][f]), p[f"{f}_min"], p[f"{f}_max"]) for f in FEATURES]
            hid = [sigmoid(b1[h] + sum(si[j] * w1[j][h] for j in range(INPUT_SIZE))) for h in range(HIDDEN_SIZE)]
            out = sigmoid(b2 + sum(hid[h] * w2[h] for h in range(HIDDEN_SIZE)))
            ann_preds.append(round(unscale(out, tmin, tmax), 2))
    return lr_preds, ann_preds

# ---------------------------------------------------
# BASE CONTEXT (shared across all pages)
# ---------------------------------------------------
def base_ctx():
    return {
        "companies":  get_companies(),
        "is_admin":   session.get("is_admin", False),
        "site_name":  "ShareSathi",
        "site_tag":   "सेयर साथी",
    }

# ---------------------------------------------------
# USER ROUTES
# ---------------------------------------------------
@app.route("/")
def index():
    return serve_page("index.html")

@app.route("/sentiment")
def sentiment_page():
    return serve_page("sentiment.html")

@app.route("/predict")
def predict():
    return serve_page("predict.html")

@app.route("/companies")
def companies_page():
    return serve_page("companies.html")

@app.route("/api/session")
def api_session():
    return jsonify({"success": True, "is_admin": bool(session.get("is_admin", False))})

@app.route("/api/companies")
def api_companies():
    return jsonify({"success": True, "companies": get_companies(), "categories": CATEGORIZED_COMPANIES})

@app.route("/api/sentiment-data")
def api_sentiment_data():
    sentiment_data = get_sentiment_data()
    rows = []
    for sym, score in sentiment_data.items():
        rows.append({"symbol": sym, "score": score})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return jsonify({"success": True, "rows": rows, "categories": CATEGORIZED_COMPANIES})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    company = request.form.get("company")
    if not company:
        return jsonify({"success": False, "error": "Please select a company."}), 400

    train_path = os.path.join(merged_path, f"{company}_train.csv")
    if not os.path.exists(train_path):
        return jsonify({"success": False, "error": f"No data found for {company}."}), 404

    data = pd.read_csv(train_path)
    data = fix_columns(data)
    data = data.sort_values("date").reset_index(drop=True)

    if "ltp" not in data.columns:
        return jsonify({"success": False, "error": f"Column 'ltp' not found for {company}."}), 400

    latest = data.tail(1).iloc[0]
    linear_pred, linear_r2 = predict_linear(company, latest)
    ann_pred, ann_r2 = predict_ann(company, latest)
    chart_data = get_chart_data(company)
    lr_preds, ann_preds = get_comparison_data(company)

    ltp_val = float(latest["ltp"]) if "ltp" in data.columns else float(latest["target"])
    better_model = "ANN" if (ann_r2 or 0) >= (linear_r2 or 0) else "Linear Regression"
    diff = round(abs((ann_r2 or 0) - (linear_r2 or 0)), 2)

    return jsonify({
        "success": True,
        "company": company,
        "linear_pred": linear_pred,
        "linear_r2": linear_r2,
        "ann_pred": ann_pred,
        "ann_r2": ann_r2,
        "chart_data": chart_data,
        "latest": {
            "date": str(latest["date"]),
            "open": round(float(latest["open"]), 2),
            "high": round(float(latest["high"]), 2),
            "low": round(float(latest["low"]), 2),
            "ltp": round(ltp_val, 2),
            "qty": int(float(latest["qty"])),
            "sentiment": float(latest["sentiment"]),
        },
        "lr_preds": lr_preds,
        "ann_preds": ann_preds,
        "dates": data["date"].astype(str).tolist(),
        "better_model": better_model,
        "diff": diff,
    })

# ---------------------------------------------------
# ADMIN ROUTES
# ---------------------------------------------------
@app.route("/admin/login")
def admin_login():
    return serve_page("admin_login.html")

@app.route("/api/admin/login", methods=["POST"])
def api_admin_login():
    if request.form.get("username") == ADMIN_USER and request.form.get("password") == ADMIN_PASS:
        session["is_admin"] = True
        return jsonify({"success": True, "redirect": url_for("admin_dashboard")})
    return jsonify({"success": False, "error": "Invalid username or password."}), 401

@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("index"))

@app.route("/admin")
def admin_dashboard():
    return serve_page("admin_dashboard.html")

@app.route("/api/admin/dashboard")
def api_admin_dashboard():
    if not session.get("is_admin"):
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    sentiment_data = get_sentiment_data()
    rows = []
    for sym, score in sentiment_data.items():
        sig, cls = sentiment_signal(score)
        rows.append({"symbol": sym, "score": score, "signal": sig, "cls": cls})
    rows.sort(key=lambda r: r["score"], reverse=True)
    total_companies = len(get_companies())
    total_training = 0
    class_counts = {}
    if os.path.exists(training_csv):
        df = pd.read_csv(training_csv)
        total_training = len(df)
        class_counts = df["Sentiment"].value_counts().to_dict()
    return jsonify({
        "success": True,
        "total_companies": total_companies,
        "total_training": total_training,
        "class_counts": class_counts,
        "sentiment_rows": rows,
    })

@app.route("/admin/training")
def admin_training():
    return serve_page("admin_training.html")

@app.route("/api/admin/training", methods=["GET", "POST"])
def api_admin_training():
    if not session.get("is_admin"):
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        sentiment = request.form.get("sentiment", "").strip()
        if not text or not sentiment:
            return jsonify({"success": False, "error": "Both text and sentiment are required."}), 400
        new_row = pd.DataFrame([{"Text": text, "Sentiment": sentiment}])
        if os.path.exists(training_csv):
            df = pd.read_csv(training_csv)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row
        df.to_csv(training_csv, index=False)
        return jsonify({"success": True, "message": "Sample added."})
    if os.path.exists(training_csv):
        df = pd.read_csv(training_csv)
        rows = df.to_dict("records")
        class_counts = df["Sentiment"].value_counts().to_dict()
        total = len(df)
    else:
        rows = []
        class_counts = {}
        total = 0
    return jsonify({
        "success": True,
        "rows": rows,
        "class_counts": class_counts,
        "total": total,
        "sentiment_classes": [
            "Strong_Bullish", "Weak_Bullish", "Neutral", "Weak_Bearish", "Strong_Bearish"
        ],
    })

@app.route("/api/admin/training/delete/<int:idx>", methods=["POST"])
def api_admin_delete_training(idx):
    if not session.get("is_admin"):
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    if os.path.exists(training_csv):
        df = pd.read_csv(training_csv)
        if 0 <= idx < len(df):
            df = df.drop(index=idx).reset_index(drop=True)
            df.to_csv(training_csv, index=False)
            return jsonify({"success": True, "message": "Sample deleted."})
    return jsonify({"success": False, "error": "Index out of range."}), 400

if __name__ == "__main__":
    app.run(debug=True)
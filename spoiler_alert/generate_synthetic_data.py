import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import joblib

CATEGORIES = {
    "raw_meat": {"shelf_life": 48,  "weights": {"BCG": 0.20, "BTB": 0.60, "KMNO4": 0.20}},
    "dairy":    {"shelf_life": 144, "weights": {"BCG": 0.50, "BTB": 0.30, "KMNO4": 0.20}},
    "leafy":    {"shelf_life": 96,  "weights": {"BCG": 0.30, "BTB": 0.20, "KMNO4": 0.50}},
    "cooked":   {"shelf_life": 84,  "weights": {"BCG": 0.33, "BTB": 0.33, "KMNO4": 0.34}},
}
LIFECYCLES_PER_CAT = 18
TIMEPOINTS         = 17
NOISE_STD          = 0.015
RETRAIN_THRESHOLD  = 5

def generate_lifecycle(category: str, cat_params: dict) -> list[dict]:
    # Sample total_lifetime: uniform(12, 72) hours
    total_lifetime = np.random.uniform(12, 72)
    
    k_bcg = np.random.uniform(0.05, 0.10)
    t0_bcg = total_lifetime * np.random.uniform(0.45, 0.60)
    
    k_btb = np.random.uniform(0.08, 0.15)
    t0_btb = total_lifetime * np.random.uniform(0.55, 0.70)
    
    k_kmno4 = np.random.uniform(0.12, 0.20)
    t0_kmno4 = total_lifetime * np.random.uniform(0.35, 0.50)
    
    times = np.linspace(0, total_lifetime, TIMEPOINTS)
    
    def get_normalized_scores(k, t0):
        raw = 1 / (1 + np.exp(-k * (times - t0)))
        raw_0 = 1 / (1 + np.exp(-k * (0 - t0)))
        raw_T = 1 / (1 + np.exp(-k * (total_lifetime - t0)))
        
        if raw_T == raw_0:
            norm = np.zeros_like(times)
        else:
            norm = (raw - raw_0) / (raw_T - raw_0)
            
        norm_noisy = norm + np.random.normal(0, NOISE_STD, len(times))
        return np.clip(norm_noisy, 0, 1)

    bcg_scores = get_normalized_scores(k_bcg, t0_bcg)
    btb_scores = get_normalized_scores(k_btb, t0_btb)
    kmno4_scores = get_normalized_scores(k_kmno4, t0_kmno4)
    
    readings = []
    for i, t in enumerate(times):
        readings.append({
            "time": t,
            "bcg_score": bcg_scores[i],
            "btb_score": btb_scores[i],
            "kmno4_score": kmno4_scores[i],
            "total_lifetime": total_lifetime,
            "category": category
        })
    return readings

def extract_features(readings: list[dict], current_idx: int, cat_params: dict) -> dict | None:
    if current_idx < 2:
        return None
        
    current = readings[current_idx]
    hours_remaining = current["total_lifetime"] - current["time"]
    
    if hours_remaining <= 0:
        return None
        
    hours_elapsed = current["time"]
    
    w_bcg = cat_params["weights"]["BCG"]
    w_btb = cat_params["weights"]["BTB"]
    w_kmno4 = cat_params["weights"]["KMNO4"]
    
    bcg_score = current["bcg_score"]
    btb_score = current["btb_score"]
    kmno4_score = current["kmno4_score"]
    
    current_score = (bcg_score * w_bcg) + (btb_score * w_btb) + (kmno4_score * w_kmno4)
    
    rates = []
    times_for_rates = []
    for i in range(1, current_idx + 1):
        prev = readings[i-1]
        curr = readings[i]
        
        prev_score = (prev["bcg_score"] * w_bcg) + (prev["btb_score"] * w_btb) + (prev["kmno4_score"] * w_kmno4)
        curr_score = (curr["bcg_score"] * w_bcg) + (curr["btb_score"] * w_btb) + (curr["kmno4_score"] * w_kmno4)
        
        score_diff = curr_score - prev_score
        time_diff = curr["time"] - prev["time"]
        rate = score_diff / time_diff if time_diff > 0 else 0
        rates.append(rate)
        times_for_rates.append(curr["time"])
        
    mean_rate = np.mean(rates)
    max_rate = np.max(rates)
    
    if len(rates) > 1:
        slope, _, _, _, _ = linregress(times_for_rates, rates)
        rate_trend = slope
    else:
        rate_trend = 0.0
        
    scores = [bcg_score, btb_score, kmno4_score]
    dominant_sticker = int(np.argmax(scores))
    
    cat_mapping = {"raw_meat": 0, "dairy": 1, "leafy": 2, "cooked": 3}
    food_category = cat_mapping[current["category"]]
    
    return {
        "hours_elapsed": hours_elapsed,
        "current_score": current_score,
        "mean_rate": mean_rate,
        "max_rate": max_rate,
        "rate_trend": rate_trend,
        "bcg_score": bcg_score,
        "btb_score": btb_score,
        "kmno4_score": kmno4_score,
        "dominant_sticker": dominant_sticker,
        "food_category": food_category,
        "hours_remaining": hours_remaining
    }

def main():
    np.random.seed(42)
    all_features = []
    
    for cat, cat_params in CATEGORIES.items():
        for _ in range(LIFECYCLES_PER_CAT):
            readings = generate_lifecycle(cat, cat_params)
            for i in range(len(readings)):
                feats = extract_features(readings, i, cat_params)
                if feats is not None:
                    all_features.append(feats)
                    
    df = pd.DataFrame(all_features)
    df.to_csv("synthetic_training_data.csv", index=False)
    
    print(f"Total rows: {len(df)}")
    cat_mapping_inv = {0: "raw_meat", 1: "dairy", 2: "leafy", 3: "cooked"}
    for cat_idx in df['food_category'].unique():
        count = len(df[df['food_category'] == cat_idx])
        print(f"  {cat_mapping_inv[cat_idx]}: {count} rows")
        
    print(f"hours_remaining mean: {df['hours_remaining'].mean():.2f}, std: {df['hours_remaining'].std():.2f}")
    
    feature_cols = [
        "hours_elapsed", "current_score", "mean_rate", "max_rate", "rate_trend",
        "bcg_score", "btb_score", "kmno4_score", "dominant_sticker", "food_category"
    ]
    X = df[feature_cols].values
    y = df["hours_remaining"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")
    
    model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    
    batch_size = 50
    for i in range(0, len(X_scaled), batch_size):
        X_batch = X_scaled[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        model.partial_fit(X_batch, y_batch)
        
    joblib.dump(model, "spoilage_model.pkl")
    
    y_pred = model.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"Train RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()

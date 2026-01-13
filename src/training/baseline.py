# baseline.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib
from math import sqrt

# ─── 1. Load & Merge ────────────────────────────────────────────────
print("Loading data...")
df_train = pd.read_csv("train.csv", low_memory=False, parse_dates=["Date"])
df_store = pd.read_csv("store.csv", low_memory=False)

print("Merging train + store...")
df = df_train.merge(df_store, on="Store", how="left")

# ─── 2. Very basic feature engineering ───────────────────────────────
print("Basic feature extraction...")
df["Year"]  = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"]   = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek + 1   # 1=Monday, 7=Sunday (matches original)

# ─── 3. Handle missing values (very naive) ───────────────────────────
print("Filling missing values...")

# Competition rule: if Store is closed → Sales = 0
df["Sales"] = df["Sales"].fillna(0)
df["Customers"] = df["Customers"].fillna(0)

# Most important columns that often have NaN
df["CompetitionDistance"] = df["CompetitionDistance"].fillna(999999)   # far away
df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(1)
df["CompetitionOpenSinceYear"]  = df["CompetitionOpenSinceYear"].fillna(1900)
df["Promo2SinceWeek"]  = df["Promo2SinceWeek"].fillna(0)
df["Promo2SinceYear"]  = df["Promo2SinceYear"].fillna(0)
df["PromoInterval"]    = df["PromoInterval"].fillna("None")

# Others → median or zero
for col in ["SchoolHoliday"]:
    df[col] = df[col].fillna(0)

# ─── 4. Encode categorical columns ──────────────────────────────────
print("Encoding categoricals...")

cat_cols = ["StoreType", "Assortment", "PromoInterval", "StateHoliday"]

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    # handle NaN safely
    df[col] = df[col].astype(str).fillna("missing")
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# StateHoliday is often '0','a','b','c' → already handled by LabelEncoder

# ─── 5. Time-based split (last ~6 weeks as validation) ───────────────
print("Creating time-based train/val split...")

df = df.sort_values("Date")

cutoff_date = "2015-06-19"          # roughly last 6 weeks of train set
# or more precisely: last 41-42 days depending on exact split you like

train = df[df["Date"] < cutoff_date].copy()
valid = df[df["Date"] >= cutoff_date].copy()

print(f"Train shape: {train.shape} | Valid shape: {valid.shape}")
print(f"Validation period: {valid['Date'].min()} → {valid['Date'].max()}")

# ─── 6. Select features ──────────────────────────────────────────────
features = [
    "Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday",
    "Year", "Month", "Day",
    "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
    # "Customers"  ← usually excluded in true forecasting
]

# Important: filter only open days for training (very common in baselines)
train = train[train["Open"] == 1].copy()
valid = valid[valid["Open"] == 1].copy()   # we only evaluate on open days

X_train = train[features]
y_train = train["Sales"]

X_valid = valid[features]
y_valid = valid["Sales"]

print(f"Train samples (open days): {len(X_train):,}")
print(f"Valid samples (open days): {len(X_valid):,}")

# ─── 7. Train baseline model ─────────────────────────────────────────
print("Training RandomForestRegressor...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=35,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

# ─── 8. Predict & Evaluate ───────────────────────────────────────────
print("Predicting on validation set...")
y_pred = model.predict(X_valid)

rmse = sqrt(mean_squared_error(y_valid, y_pred))
print(f"\nValidation RMSE: {rmse:,.2f}")

# Optional: RMSPE (common in Rossmann)
def rmspe(y_true, y_pred):
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true - y_pred)[mask] / y_true[mask]) ** 2)) * 100

rmspe_val = rmspe(y_valid.values, y_pred)
print(f"Validation RMSPE: {rmspe_val:.3f}%")

# ─── 9. Save model ───────────────────────────────────────────────────
print("Saving model...")
joblib.dump(model, "baseline_rf_rossmann.pkl")
joblib.dump(le_dict, "label_encoders.pkl")   # if you want to reuse later

print("Done! 🎯")
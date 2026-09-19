import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

panel = pd.read_csv('/home/claude/data/panel_a_exact.csv', parse_dates=['Week'])

FEATURES = ["qty_lag1", "qty_lag2", "qty_lag4", "qty_roll4", "qty_roll8",
            "Price", "price_rel", "price_change", "invoices_lag1",
            "week_of_year", "month", "is_q4"]
TARGET = "Quantity"
RANDOM_STATE = 42

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

cutoff = panel["Week"].quantile(0.8)
train = panel[panel["Week"] <= cutoff]
test = panel[panel["Week"] > cutoff]
print(f"[A] split at {cutoff.date()}: train={len(train):,} test={len(test):,}")

X_tr, y_tr = train[FEATURES], train[TARGET]
X_te, y_te = test[FEATURES], test[TARGET]

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, n_jobs=-1,
        random_state=RANDOM_STATE),
    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, min_samples_leaf=20,
        random_state=RANDOM_STATE),
}

rows = []
rows.append({"Model": "Naive (lag-1)",
             "MAE": mean_absolute_error(y_te, X_te["qty_lag1"]),
             "RMSE": np.sqrt(mean_squared_error(y_te, X_te["qty_lag1"])),
             "WAPE": wape(y_te, X_te["qty_lag1"])})

for name, model in models.items():
    model.fit(X_tr, y_tr)
    pred = np.clip(model.predict(X_te), 0, None)
    rows.append({"Model": name,
                 "MAE": mean_absolute_error(y_te, pred),
                 "RMSE": np.sqrt(mean_squared_error(y_te, pred)),
                 "WAPE": wape(y_te, pred)})

results = pd.DataFrame(rows).set_index("Model").round(3)
print("\n[A] DEMAND FORECASTING RESULTS (exact reproduction)\n", results, "\n")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, average_precision_score, roc_auc_score

# ---------- Dataset A: TimeSeriesSplit vs naive random KFold ----------
panel = pd.read_csv('/home/claude/data/panel_a.csv', parse_dates=['Week'])
features = ['lag1','lag2','lag4','roll4','roll8','Price','price_rel','price_chg',
            'invoices_lag1','month','weekofyear']
panel = panel.sort_values('Week').reset_index(drop=True)
X, y = panel[features], panel['Quantity']

def wape(a, p):
    return np.abs(a - p).sum() / np.abs(a).sum()

print("=== Dataset A: naive random 5-fold CV (leaks future into past) ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for tr_idx, va_idx in kf.split(X):
    m = HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05, random_state=42)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = m.predict(X.iloc[va_idx])
    scores.append(wape(y.iloc[va_idx].values, pred))
print("Random KFold WAPE per fold:", [round(s,3) for s in scores], "mean", round(np.mean(scores),3))

print("\n=== Dataset A: TimeSeriesSplit 5-fold CV (respects chronology) ===")
tscv = TimeSeriesSplit(n_splits=5)
scores_ts = []
for tr_idx, va_idx in tscv.split(X):
    m = HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05, random_state=42)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = m.predict(X.iloc[va_idx])
    scores_ts.append(wape(y.iloc[va_idx].values, pred))
print("TimeSeriesSplit WAPE per fold:", [round(s,3) for s in scores_ts], "mean", round(np.mean(scores_ts),3))

# ---------- Dataset B: StratifiedKFold ----------
df = pd.read_csv('/home/claude/data/shoppers/online_shoppers_intention.csv')
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)
df = pd.get_dummies(df, columns=['Month','VisitorType'], drop_first=False)
yb = df['Revenue']; Xb = df.drop(columns=['Revenue'])

print("\n=== Dataset B: plain KFold (ignores 15.5% class imbalance) ===")
kf2 = KFold(n_splits=5, shuffle=True, random_state=42)
pos_rates = []
pr_scores = []
for tr_idx, va_idx in kf2.split(Xb):
    pos_rates.append(yb.iloc[va_idx].mean())
    m = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.05, random_state=42)
    m.fit(Xb.iloc[tr_idx], yb.iloc[tr_idx])
    proba = m.predict_proba(Xb.iloc[va_idx])[:,1]
    pr_scores.append(average_precision_score(yb.iloc[va_idx], proba))
print("Validation-fold positive rate (%):", [round(p*100,1) for p in pos_rates])
print("PR-AUC per fold:", [round(s,3) for s in pr_scores], "mean", round(np.mean(pr_scores),3), "std", round(np.std(pr_scores),3))

print("\n=== Dataset B: StratifiedKFold (preserves 15.5% class balance each fold) ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pos_rates_s = []
pr_scores_s = []
for tr_idx, va_idx in skf.split(Xb, yb):
    pos_rates_s.append(yb.iloc[va_idx].mean())
    m = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.05, random_state=42)
    m.fit(Xb.iloc[tr_idx], yb.iloc[tr_idx])
    proba = m.predict_proba(Xb.iloc[va_idx])[:,1]
    pr_scores_s.append(average_precision_score(yb.iloc[va_idx], proba))
print("Validation-fold positive rate (%):", [round(p*100,1) for p in pos_rates_s])
print("PR-AUC per fold:", [round(s,3) for s in pr_scores_s], "mean", round(np.mean(pr_scores_s),3), "std", round(np.std(pr_scores_s),3))

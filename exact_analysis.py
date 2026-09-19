import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_absolute_error

panel = pd.read_csv('/home/claude/data/panel_a_exact.csv', parse_dates=['Week'])
FEATURES = ["qty_lag1", "qty_lag2", "qty_lag4", "qty_roll4", "qty_roll8",
            "Price", "price_rel", "price_change", "invoices_lag1",
            "week_of_year", "month", "is_q4"]
TARGET = "Quantity"
RANDOM_STATE = 42

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def make_gb():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          min_samples_leaf=20, random_state=RANDOM_STATE)

panel_sorted = panel.sort_values('Week').reset_index(drop=True)
X, y = panel_sorted[FEATURES], panel_sorted[TARGET]

print("=== CV comparison (exact features/model) ===")
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores_rand = []
for tr_idx, va_idx in kf.split(X):
    m = make_gb()
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = np.clip(m.predict(X.iloc[va_idx]), 0, None)
    scores_rand.append(wape(y.iloc[va_idx].values, pred))
print("Random KFold WAPE per fold:", [round(s,3) for s in scores_rand], "mean", round(np.mean(scores_rand),3))

tscv = TimeSeriesSplit(n_splits=5)
scores_ts = []
for tr_idx, va_idx in tscv.split(X):
    m = make_gb()
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = np.clip(m.predict(X.iloc[va_idx]), 0, None)
    scores_ts.append(wape(y.iloc[va_idx].values, pred))
print("TimeSeriesSplit WAPE per fold:", [round(s,3) for s in scores_ts], "mean", round(np.mean(scores_ts),3))

# ---------- Learning curve ----------
cutoff = panel["Week"].quantile(0.8)
train = panel[panel["Week"] <= cutoff].sort_values('Week')
test = panel[panel["Week"] > cutoff]
Xte, yte = test[FEATURES], test[TARGET]

fracs = np.linspace(0.1, 1.0, 10)
sizes, wapes = [], []
n = len(train)
for f in fracs:
    k = int(n * f)
    sub = train.iloc[-k:]
    m = make_gb()
    m.fit(sub[FEATURES], sub[TARGET])
    pred = np.clip(m.predict(Xte), 0, None)
    sizes.append(k)
    wapes.append(wape(yte.values, pred))

print("\n=== Learning curve (exact pipeline) ===")
for k, w in zip(sizes, wapes):
    print(f"  n={k:6d}  WAPE={w:.3f}")

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(sizes, wapes, marker='o', color='#2563eb')
ax.set_xlabel('Training set size (product-weeks)')
ax.set_ylabel('WAPE on held-out weeks (lower is better)')
ax.set_title('Dataset A — Gradient Boosting learning curve (exact pipeline)')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('/home/claude/figures/learning_curve_a_exact.png', dpi=150)
plt.close(fig)

# ---------- Bias by revenue tier ----------
m = make_gb()
m.fit(train[FEATURES], train[TARGET])
test = test.copy()
test['pred'] = np.clip(m.predict(Xte), 0, None)
test['abs_err'] = (test['Quantity'] - test['pred']).abs()

rev = train.assign(rev=train['Quantity']*train['Price']).groupby('StockCode')['rev'].sum()
tiers = pd.qcut(rev, 3, labels=['Low (bottom third)','Mid (middle third)','High (top third)'])
test['tier'] = test['StockCode'].map(tiers)

group_a = test.groupby('tier', observed=True).apply(
    lambda d: pd.Series({
        'n_obs': len(d),
        'MAE': mean_absolute_error(d['Quantity'], d['pred']),
        'WAPE': d['abs_err'].sum() / d['Quantity'].abs().sum() if d['Quantity'].abs().sum() > 0 else np.nan,
        'mean_actual_demand': d['Quantity'].mean(),
    })
)
print("\n=== Bias by product revenue tier (exact pipeline) ===")
print(group_a)
group_a.to_csv('/home/claude/data/bias_a_tier_exact.csv')

fig, ax = plt.subplots(figsize=(7,5))
group_a['WAPE'].plot(kind='bar', ax=ax, color=['#f97316','#3b82f6','#16a34a'])
ax.set_ylabel('WAPE (lower is better)')
ax.set_title('Dataset A — forecast error by product revenue tier (exact pipeline)')
ax.set_xlabel('')
plt.xticks(rotation=20)
fig.tight_layout()
fig.savefig('/home/claude/figures/bias_a_tier_exact.png', dpi=150)
plt.close(fig)

print("\nSaved figures.")

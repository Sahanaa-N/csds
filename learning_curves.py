import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, average_precision_score

# ---------- Dataset A: learning curve (temporal train, fixed temporal test) ----------
panel = pd.read_csv('/home/claude/data/panel_a.csv', parse_dates=['Week'])
features = ['lag1','lag2','lag4','roll4','roll8','Price','price_rel','price_chg',
            'invoices_lag1','month','weekofyear']
split_date = pd.Timestamp('2011-07-25')
train = panel[panel['Week'] < split_date].sort_values('Week')
test = panel[panel['Week'] >= split_date]
Xte, yte = test[features], test['Quantity']

def wape(a, p):
    return np.abs(a - p).sum() / np.abs(a).sum()

fracs = np.linspace(0.1, 1.0, 10)
train_sizes_a, wape_scores_a, mae_scores_a = [], [], []
n = len(train)
for f in fracs:
    k = int(n * f)
    sub = train.iloc[-k:]  # most recent k rows, preserving temporal adjacency to test
    Xtr, ytr = sub[features], sub['Quantity']
    m = HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05, random_state=42)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    train_sizes_a.append(k)
    wape_scores_a.append(wape(yte.values, pred))
    mae_scores_a.append(mean_absolute_error(yte, pred))

print("Dataset A learning curve (train size -> WAPE):")
for k, w in zip(train_sizes_a, wape_scores_a):
    print(f"  n={k:6d}  WAPE={w:.3f}")

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(train_sizes_a, wape_scores_a, marker='o', color='#2563eb')
ax.set_xlabel('Training set size (product-weeks)')
ax.set_ylabel('WAPE on held-out weeks (lower is better)')
ax.set_title('Dataset A -- Gradient Boosting learning curve')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('/home/claude/figures/learning_curve_a.png', dpi=150)
plt.close(fig)

# ---------- Dataset B: learning curve (stratified subsampling of train) ----------
from sklearn.model_selection import train_test_split
df = pd.read_csv('/home/claude/data/shoppers/online_shoppers_intention.csv')
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)
df = pd.get_dummies(df, columns=['Month','VisitorType'], drop_first=False)
y = df['Revenue']; X = df.drop(columns=['Revenue'])
Xtr_full, Xte_b, ytr_full, yte_b = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

fracs_b = np.linspace(0.1, 1.0, 10)
train_sizes_b, prauc_scores_b = [], []
for f in fracs_b:
    if f < 1.0:
        Xsub, _, ysub, _ = train_test_split(Xtr_full, ytr_full, train_size=f, stratify=ytr_full, random_state=42)
    else:
        Xsub, ysub = Xtr_full, ytr_full
    m = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.05, random_state=42)
    m.fit(Xsub, ysub)
    proba = m.predict_proba(Xte_b)[:,1]
    train_sizes_b.append(len(Xsub))
    prauc_scores_b.append(average_precision_score(yte_b, proba))

print("\nDataset B learning curve (train size -> PR-AUC):")
for k, s in zip(train_sizes_b, prauc_scores_b):
    print(f"  n={k:6d}  PR-AUC={s:.3f}")

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(train_sizes_b, prauc_scores_b, marker='o', color='#16a34a')
ax.set_xlabel('Training set size (sessions)')
ax.set_ylabel('PR-AUC on held-out sessions (higher is better)')
ax.set_title('Dataset B -- Gradient Boosting learning curve')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('/home/claude/figures/learning_curve_b.png', dpi=150)
plt.close(fig)

print("\nSaved figures.")

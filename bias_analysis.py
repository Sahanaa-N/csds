import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ================= Dataset A: error by product revenue tier =================
panel = pd.read_csv('/home/claude/data/panel_a.csv', parse_dates=['Week'])
features = ['lag1','lag2','lag4','roll4','roll8','Price','price_rel','price_chg',
            'invoices_lag1','month','weekofyear']
split_date = pd.Timestamp('2011-07-25')
train = panel[panel['Week'] < split_date]
test = panel[panel['Week'] >= split_date].copy()

m = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42)
m.fit(train[features], train['Quantity'])
test['pred'] = m.predict(test[features])
test['abs_err'] = (test['Quantity'] - test['pred']).abs()

# revenue tier: total training-period revenue per product, split into thirds
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
print("=== Dataset A: disaggregated error by product revenue tier ===")
print(group_a)
group_a.to_csv('/home/claude/data/bias_a_by_tier.csv')

fig, ax = plt.subplots(figsize=(7,5))
group_a['WAPE'].plot(kind='bar', ax=ax, color=['#f97316','#3b82f6','#16a34a'])
ax.set_ylabel('WAPE (lower is better)')
ax.set_title('Dataset A -- forecast error by product revenue tier')
ax.set_xlabel('')
plt.xticks(rotation=20)
fig.tight_layout()
fig.savefig('/home/claude/figures/bias_a_tier.png', dpi=150)
plt.close(fig)

# ================= Dataset B: error rates by Region and VisitorType =================
df = pd.read_csv('/home/claude/data/shoppers/online_shoppers_intention.csv')
df_raw = df.copy()
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)
df_enc = pd.get_dummies(df, columns=['Month','VisitorType'], drop_first=False)
y = df_enc['Revenue']; X = df_enc.drop(columns=['Revenue'])

Xtr, Xte, ytr, yte, raw_tr, raw_te = train_test_split(
    X, y, df_raw, test_size=0.25, stratify=y, random_state=42)

m2 = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42)
m2.fit(Xtr, ytr)
proba = m2.predict_proba(Xte)[:, 1]
pred = (proba >= 0.35).astype(int)

raw_te = raw_te.copy()
raw_te['y_true'] = yte.values
raw_te['y_pred'] = pred
raw_te['proba'] = proba

def group_metrics(d):
    tp = ((d['y_true']==1) & (d['y_pred']==1)).sum()
    fn = ((d['y_true']==1) & (d['y_pred']==0)).sum()
    fp = ((d['y_true']==0) & (d['y_pred']==1)).sum()
    tn = ((d['y_true']==0) & (d['y_pred']==0)).sum()
    n_pos = tp + fn
    n_neg = fp + tn
    return pd.Series({
        'n_sessions': len(d),
        'actual_conv_rate': d['y_true'].mean(),
        'selection_rate': d['y_pred'].mean(),
        'FNR': fn / n_pos if n_pos > 0 else np.nan,   # missed converters
        'FPR': fp / n_neg if n_neg > 0 else np.nan,
    })

print("\n=== Dataset B: disaggregated error by VisitorType ===")
grp_visitor = raw_te.groupby('VisitorType').apply(group_metrics)
print(grp_visitor)
grp_visitor.to_csv('/home/claude/data/bias_b_by_visitor.csv')

print("\n=== Dataset B: disaggregated error by Region ===")
grp_region = raw_te.groupby('Region').apply(group_metrics)
print(grp_region)
grp_region.to_csv('/home/claude/data/bias_b_by_region.csv')

fig, axes = plt.subplots(1, 2, figsize=(11,5))
grp_visitor['FNR'].plot(kind='bar', ax=axes[0], color='#dc2626')
axes[0].set_title('False Negative Rate by Visitor Type\n(converting sessions missed by the model)')
axes[0].set_ylabel('FNR (lower is better)')
axes[0].set_xlabel('')
plt.setp(axes[0].get_xticklabels(), rotation=20)

grp_region['FNR'].plot(kind='bar', ax=axes[1], color='#7c3aed')
axes[1].set_title('False Negative Rate by Region')
axes[1].set_ylabel('FNR (lower is better)')
axes[1].set_xlabel('Region code')
fig.tight_layout()
fig.savefig('/home/claude/figures/bias_b_fnr.png', dpi=150)
plt.close(fig)

print("\nSaved figures.")

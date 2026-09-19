import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

df = pd.read_csv('/home/claude/data/shoppers/online_shoppers_intention.csv')
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)
df_enc = pd.get_dummies(df, columns=['Month','VisitorType'], drop_first=False)
yb = df_enc['Revenue']; Xb = df_enc.drop(columns=['Revenue'])

print("=== Dataset B: plain KFold (ignores 15.5% class imbalance) ===")
kf2 = KFold(n_splits=5, shuffle=True, random_state=42)
pos_rates, pr_scores = [], []
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
pos_rates_s, pr_scores_s = [], []
for tr_idx, va_idx in skf.split(Xb, yb):
    pos_rates_s.append(yb.iloc[va_idx].mean())
    m = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.05, random_state=42)
    m.fit(Xb.iloc[tr_idx], yb.iloc[tr_idx])
    proba = m.predict_proba(Xb.iloc[va_idx])[:,1]
    pr_scores_s.append(average_precision_score(yb.iloc[va_idx], proba))
print("Validation-fold positive rate (%):", [round(p*100,1) for p in pos_rates_s])
print("PR-AUC per fold:", [round(s,3) for s in pr_scores_s], "mean", round(np.mean(pr_scores_s),3), "std", round(np.std(pr_scores_s),3))

# ---- Bias by VisitorType and Region ----
df_raw = pd.read_csv('/home/claude/data/shoppers/online_shoppers_intention.csv')
Xtr, Xte, ytr, yte, raw_tr, raw_te = train_test_split(
    Xb, yb, df_raw, test_size=0.25, stratify=yb, random_state=42)

m2 = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42)
m2.fit(Xtr, ytr)
proba = m2.predict_proba(Xte)[:, 1]
pred = (proba >= 0.35).astype(int)

raw_te = raw_te.copy()
raw_te['y_true'] = yte.values
raw_te['y_pred'] = pred

def group_metrics(d):
    tp = ((d['y_true']==1) & (d['y_pred']==1)).sum()
    fn = ((d['y_true']==1) & (d['y_pred']==0)).sum()
    fp = ((d['y_true']==0) & (d['y_pred']==1)).sum()
    tn = ((d['y_true']==0) & (d['y_pred']==0)).sum()
    n_pos = tp + fn; n_neg = fp + tn
    return pd.Series({
        'n_sessions': len(d),
        'actual_conv_rate': d['y_true'].mean(),
        'selection_rate': d['y_pred'].mean(),
        'FNR': fn / n_pos if n_pos > 0 else np.nan,
        'FPR': fp / n_neg if n_neg > 0 else np.nan,
    })

print("\n=== Dataset B: disaggregated error by VisitorType ===")
print(raw_te.groupby('VisitorType').apply(group_metrics))

print("\n=== Dataset B: disaggregated error by Region ===")
print(raw_te.groupby('Region').apply(group_metrics))

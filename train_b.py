import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve

df = pd.read_csv('/home/claude/data/shoppers/online_shoppers_intention.csv')
print(df.shape)

df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)
df = pd.get_dummies(df, columns=['Month','VisitorType'], drop_first=False)

y = df['Revenue']
X = df.drop(columns=['Revenue'])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print('train', Xtr.shape, 'test', Xte.shape, 'pos rate train', ytr.mean(), 'test', yte.mean())

def eval_model(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pr_auc = average_precision_score(yte, proba)
    roc_auc = roc_auc_score(yte, proba)
    # tune threshold for F1
    prec, rec, thr = precision_recall_curve(yte, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.nanargmax(f1s)
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
    f1 = f1_score(yte, (proba >= best_thr).astype(int))
    print(f'{name:10s} PR-AUC {pr_auc:.3f} ROC-AUC {roc_auc:.3f} F1 {f1:.3f} (thr={best_thr:.3f})')
    return model, proba, best_thr

rf, rf_proba, rf_thr = eval_model('RF', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1), Xtr, ytr, Xte, yte)
gb, gb_proba, gb_thr = eval_model('GB', HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42), Xtr, ytr, Xte, yte)

always_no = np.zeros(len(yte))
print('Always-no  PR-AUC %.3f ROC-AUC %.3f F1 %.3f' % (
    average_precision_score(yte, always_no + yte.mean()),  # trivial proxy
    0.5, 0.0))

import joblib
joblib.dump({'rf': rf, 'gb': gb, 'Xtr': Xtr, 'ytr': ytr, 'Xte': Xte, 'yte': yte,
             'rf_proba': rf_proba, 'gb_proba': gb_proba, 'rf_thr': rf_thr, 'gb_thr': gb_thr},
            '/home/claude/data/models_b.pkl')

# keep original (non-dummy) columns for fairness grouping later
df_raw = pd.read_csv('/home/claude/data/shoppers/online_shoppers_intention.csv')
df_raw.to_csv('/home/claude/data/shoppers_raw_ref.csv', index=False)
Xte.index.to_series().to_csv('/home/claude/data/test_index_b.csv', index=False)

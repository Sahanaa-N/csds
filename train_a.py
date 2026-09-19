import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

panel = pd.read_csv('/home/claude/data/panel_a.csv', parse_dates=['Week'])

features = ['lag1','lag2','lag4','roll4','roll8','Price','price_rel','price_chg',
            'invoices_lag1','month','weekofyear']
target = 'Quantity'

split_date = pd.Timestamp('2011-07-25')
train = panel[panel['Week'] < split_date]
test = panel[panel['Week'] >= split_date]
print('train', train.shape, 'test', test.shape)

Xtr, ytr = train[features], train[target]
Xte, yte = test[features], test[target]

def wape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

# Naive lag-1 baseline
naive_pred = Xte['lag1'].values
print('Naive  MAE %.2f RMSE %.2f WAPE %.3f' % (
    mean_absolute_error(yte, naive_pred),
    mean_squared_error(yte, naive_pred) ** 0.5,
    wape(yte.values, naive_pred)))

rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(Xtr, ytr)
rf_pred = rf.predict(Xte)
print('RF     MAE %.2f RMSE %.2f WAPE %.3f' % (
    mean_absolute_error(yte, rf_pred),
    mean_squared_error(yte, rf_pred) ** 0.5,
    wape(yte.values, rf_pred)))

gb = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42)
gb.fit(Xtr, ytr)
gb_pred = gb.predict(Xte)
print('GB     MAE %.2f RMSE %.2f WAPE %.3f' % (
    mean_absolute_error(yte, gb_pred),
    mean_squared_error(yte, gb_pred) ** 0.5,
    wape(yte.values, gb_pred)))

# save for later reuse
import joblib
joblib.dump({'rf': rf, 'gb': gb, 'features': features, 'train': train, 'test': test}, '/home/claude/data/models_a.pkl')

import pandas as pd
import numpy as np

df = pd.read_csv('/home/claude/data/retail/combined.csv', parse_dates=['InvoiceDate'])
print('raw', df.shape)

# 1. Remove cancellations (Invoice starting with C)
df = df[~df['Invoice'].astype(str).str.startswith('C')]

# 2. Remove negative quantities, non-positive prices
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

# 3. Remove non-product line items (postage, manual adjustments, fees, test codes, gift cards)
exclude_codes = {'POST','DOT','M','D','C2','C3','BANK CHARGES','ADJUST','ADJUST2',
                  'AMAZONFEE','CRUK','PADS','S','B','GIFT','TEST001','TEST002','SP1002'}
df['StockCode'] = df['StockCode'].astype(str)
df = df[~df['StockCode'].isin(exclude_codes)]
df = df[~df['StockCode'].str.startswith('gift_', na=False)]

print('cleaned', df.shape, f"retained {df.shape[0]/1067371:.3%}")

# 4. Top 300 products by revenue
df['Revenue'] = df['Quantity'] * df['Price']
top300 = df.groupby('StockCode')['Revenue'].sum().nlargest(300).index
dfa = df[df['StockCode'].isin(top300)].copy()

# 5. Build weekly panel
dfa['Week'] = dfa['InvoiceDate'].dt.to_period('W-MON').dt.start_time
weeks = pd.date_range(dfa['Week'].min(), dfa['Week'].max(), freq='7D')
print('n weeks', len(weeks))

agg = dfa.groupby(['StockCode','Week']).agg(
    Quantity=('Quantity','sum'),
    Price=('Price','mean'),
    Invoices=('Invoice','nunique'),
).reset_index()

# full product x week grid, zero-fill missing weeks
idx = pd.MultiIndex.from_product([top300, weeks], names=['StockCode','Week'])
panel = agg.set_index(['StockCode','Week']).reindex(idx).reset_index()
panel['Quantity'] = panel['Quantity'].fillna(0)
panel['Invoices'] = panel['Invoices'].fillna(0)
panel = panel.sort_values(['StockCode','Week'])
panel['Price'] = panel.groupby('StockCode')['Price'].ffill().bfill()

print('panel shape', panel.shape)

# 6. Feature engineering
g = panel.groupby('StockCode')
panel['lag1'] = g['Quantity'].shift(1)
panel['lag2'] = g['Quantity'].shift(2)
panel['lag4'] = g['Quantity'].shift(4)
panel['roll4'] = g['Quantity'].shift(1).rolling(4).mean().reset_index(level=0, drop=True)
panel['roll8'] = g['Quantity'].shift(1).rolling(8).mean().reset_index(level=0, drop=True)
panel['price_med'] = g['Price'].transform('median')
panel['price_rel'] = panel['Price'] / panel['price_med']
panel['price_chg'] = panel.groupby('StockCode')['Price'].pct_change().fillna(0)
panel['invoices_lag1'] = g['Invoices'].shift(1)
panel['month'] = panel['Week'].dt.month
panel['weekofyear'] = panel['Week'].dt.isocalendar().week.astype(int)

panel = panel.dropna(subset=['lag1','lag2','lag4','roll4','roll8','invoices_lag1'])
print('after lag construction', panel.shape)

panel.to_csv('/home/claude/data/panel_a.csv', index=False)
print(panel[['StockCode','Week','Quantity','lag1','roll4','Price']].head())

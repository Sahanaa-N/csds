import pandas as pd
import numpy as np

retail_raw = pd.read_csv('/home/claude/data/retail/combined.csv', parse_dates=['InvoiceDate'])
retail_raw = retail_raw.rename(columns={'Customer ID': 'CustomerID'})
print("retail:", retail_raw.shape)

def clean_retail(df):
    n0 = len(df)
    df = df.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df = df.dropna(subset=["StockCode", "InvoiceDate"])
    df["StockCode"] = df["StockCode"].astype(str).str.upper().str.strip()
    non_product = {"POST", "D", "DOT", "M", "S", "AMAZONFEE", "BANK CHARGES",
                   "C2", "CRUK", "PADS", "B", "GIFT"}
    df = df[~df["StockCode"].isin(non_product)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["Price"]
    print(f"[A] cleaning: {n0:,} -> {len(df):,} rows "
          f"({100*(n0-len(df))/n0:.1f}% removed)")
    return df

def build_weekly_panel(df, top_n=300):
    top = (df.groupby("StockCode")["Revenue"].sum()
             .nlargest(top_n).index)
    d = df[df["StockCode"].isin(top)].copy()
    d["Week"] = d["InvoiceDate"].dt.to_period("W").dt.start_time

    panel = (d.groupby(["StockCode", "Week"])
               .agg(Quantity=("Quantity", "sum"),
                    Price=("Price", "mean"),
                    Invoices=("Invoice", "nunique"))
               .reset_index())

    weeks = pd.date_range(panel["Week"].min(), panel["Week"].max(), freq="W-MON")
    grid = pd.MultiIndex.from_product(
        [panel["StockCode"].unique(), weeks], names=["StockCode", "Week"]
    ).to_frame(index=False)
    panel = grid.merge(panel, on=["StockCode", "Week"], how="left")
    panel["Quantity"] = panel["Quantity"].fillna(0)
    panel["Invoices"] = panel["Invoices"].fillna(0)
    panel = panel.sort_values(["StockCode", "Week"])

    panel["Price"] = panel.groupby("StockCode")["Price"].ffill()
    panel["Price"] = panel.groupby("StockCode")["Price"].bfill()
    panel = panel.dropna(subset=["Price"])
    print(f"[A] panel: {panel['StockCode'].nunique()} products x "
          f"{panel['Week'].nunique()} weeks = {len(panel):,} rows")
    return panel

def engineer_demand_features(panel):
    p = panel.sort_values(["StockCode", "Week"]).copy()
    g = p.groupby("StockCode")

    for lag in (1, 2, 4):
        p[f"qty_lag{lag}"] = g["Quantity"].shift(lag)
    p["qty_roll4"] = g["Quantity"].shift(1).rolling(4).mean().reset_index(0, drop=True)
    p["qty_roll8"] = g["Quantity"].shift(1).rolling(8).mean().reset_index(0, drop=True)
    p["price_lag1"] = g["Price"].shift(1)

    p["invoices_lag1"] = g["Invoices"].shift(1)

    p["price_median"] = g["Price"].transform("median")
    p["price_rel"] = p["Price"] / p["price_median"]
    p["price_change"] = (p["Price"] - p["price_lag1"]) / p["price_lag1"]

    p["week_of_year"] = p["Week"].dt.isocalendar().week.astype(int)
    p["month"] = p["Week"].dt.month
    p["is_q4"] = p["month"].isin([10, 11, 12]).astype(int)

    p = p.dropna(subset=["qty_lag1", "qty_lag2", "qty_lag4", "qty_roll4",
                         "qty_roll8", "price_change", "invoices_lag1"])
    p = p.replace([np.inf, -np.inf], np.nan).dropna()
    return p

retail = clean_retail(retail_raw)
panel_raw = build_weekly_panel(retail, top_n=300)
panel = engineer_demand_features(panel_raw)
print(panel.shape)

panel.to_csv('/home/claude/data/panel_a_exact.csv', index=False)

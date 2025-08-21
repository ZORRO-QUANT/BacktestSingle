from pathlib import Path

import pandas as pd

from backtest.utils import save_to_excel

path_alpha = Path(
    "/Users/alan/Binance/spot/Alphas/1d/volatility_1d/vol_residualbtc_30>vol_residualbtc.csv"
)
df_alpha = pd.read_csv(path_alpha)
df_alpha["time"] = pd.to_datetime(df_alpha["time"])

path_group = Path("/Users/alan/Binance/spot/Groups/1d/amount_quarter_spot_3.csv")
df_group = pd.read_csv(path_group)
df_group["time"] = pd.to_datetime(df_group["time"])
df_group["time"] = df_group["time"] + pd.Timedelta(days=1)

df = df_group.merge(df_alpha, on=["time", "symbol"], how="left")
df.dropna(inplace=True)

dd = df.groupby("class")["vol_residualbtc_30"].mean()

print("debug")

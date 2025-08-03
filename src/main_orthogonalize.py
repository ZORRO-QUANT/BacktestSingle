import asyncio
import datetime

from backtest import OrthogonalAnalyzer
from backtest.constants import *

if __name__ == "__main__":
    # ------------------------------------
    # here we specify the start and end of the klines, NOT the alphas
    start = datetime.datetime(2022, 1, 1, 8, 0, 0)
    end = datetime.datetime(2025, 2, 20, 8, 0, 0)

    # ------------------------------------
    # state all the kline infos we want compute
    symbols = None

    groupby = GroupBy.amount_quarter_spot_3
    chunk_size = 10
    backtest_periods = (1, 5, 7, 15, 30)

    by_group = True

    # aggregations = {"STD": [6, 12, 24, 48], "MA": [6, 12, 24, 48]}
    # aggregations = {"STD": [5, 10, 15, 20], "MA": [3, 4, 5, 6, 10, 15, 20, 25, 30]}
    aggregations = {}

    # ------------------------------------
    # state all the alpha categories we want to compute
    categories = [
        Category.liquidity_1d,
    ]

    data_sources = DataSources(
        factor=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
        kline=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
        group=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
    )

    orthogonalAnalyzer = OrthogonalAnalyzer(
        start=start,
        end=end,
        symbols=symbols,
        categories=categories,
        groupby=groupby,
        chunk_size=chunk_size,
        by_group=by_group,
        backtest_periods=backtest_periods,
    )

    asyncio.run(
        orthogonalAnalyzer.compute(data_sources=data_sources, aggregations=aggregations)
    )

import datetime
import asyncio

from backtest import CategoryAnalyzer
from backtest.constants import *

if __name__ == "__main__":
    # ------------------------------------
    # here we specify the start and end of the klines, NOT the alphas
    start = datetime.datetime(2021, 1, 1, 8, 0, 0)
    end = datetime.datetime(2025, 6, 14, 8, 0, 0)

    # ------------------------------------
    # state all the kline infos we want compute
    symbols = None

    groupby = GroupBy.amount_quarter_spot_3
    chunk_size = 10
    backtest_periods = (1, 3, 5, 7, 11, 13, 15)

    by_group = True

    # aggregations = {"STD": [6, 12, 24, 48], "MA": [6, 12, 24, 48]}
    aggregations = {"STD": [5, 10, 15, 20], "MA": [3, 4, 5, 6, 10, 15, 20, 25, 30]}
    # aggregations = {}

    # ------------------------------------
    # state all the alpha categories we want to compute
    categories = [
        Category.development_1d,
    ]

    data_sources = DataSources(
        factor=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
        kline=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
        group=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
    )

    categoryAnalyzer = CategoryAnalyzer(
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
        categoryAnalyzer.compute(data_sources=data_sources, aggregations=aggregations)
    )

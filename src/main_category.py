import asyncio
import datetime

from backtest import CategoryAnalyzer
from backtest.constants import *

if __name__ == "__main__":
    # ------------------------------------
    # here we specify the start and end of the klines, NOT the alphas
    start = datetime.datetime(2022, 1, 1, 8, 0, 0)
    end = datetime.datetime(2025, 6, 30, 8, 0, 0)

    # ------------------------------------
    # state all the kline infos we want compute
    symbols = None

    groupby = GroupBy.amount_quarter_spot_3
    chunk_size = 30
    backtest_periods = (1, 3, 5, 7, 9, 15)

    by_group = True

    # aggregations = {"STD": [6, 12, 24, 48], "MA": [6, 12, 24, 48]}
    # aggregations = {"STD": [5, 10], "MA": [5, 10]}
    aggregations = {}

    # ------------------------------------
    # state all the alpha categories we want to compute
    categories = [
        # Category.liquidity_1d,
        # Category.momentum_1d,
        # Category.pv_1d,
        # Category.volatility_1d,
        Category.imbalance_1d,
        # Category.liquidity_contract,
        # Category.size_1d,
        # Category.development_1d,
        # Category.technical_1d,
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

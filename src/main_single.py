import datetime
import asyncio

from backtest import SingleAnalyzer
from backtest.constants import *

if __name__ == "__main__":
    # ------------------------------------
    # state all the kline infos we want compute
    start = datetime.datetime(2023, 1, 1, 8, 0, 0)
    end = datetime.datetime(2025, 6, 14, 8, 0, 0)

    backtest_periods = (1, 3, 5, 7, 15, 30, 60)

    # ------------------------------------
    # state all the alpha categories we want to compute
    alphas = [
        Alpha(
            category=Category.development_1d,
            alpha="fdv",
            aggregations={"MA": [5, 10, 20, 30], "STD": [5, 10, 20, 30]},
        ),
    ]

    data_sources = DataSources(
        factor=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
        kline=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
        group=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
    )

    # ------------------------------------
    # state all the necessary parameters
    by_group = True
    symbols = None
    just_metrics = True
    n_stratify = 5

    groupby = GroupBy.amount_quarter_spot_3

    # ------------------------------------
    # create the instance
    instance = SingleAnalyzer(
        groupby=groupby,
        alphas=alphas,
        symbols=symbols,
        by_group=by_group,
    )

    asyncio.run(
        instance.get_nvs_metrics(
            start=start,
            end=end,
            backtest_periods=backtest_periods,
            benchmark=Benchmark.whole,
            data_sources=data_sources,
            n_stratify=n_stratify,
            just_metrics=just_metrics,
        )
    )

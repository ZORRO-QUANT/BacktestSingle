import asyncio
import datetime

from backtest import SingleAnalyzer
from backtest.constants import *

if __name__ == "__main__":
    # ------------------------------------
    # state all the kline infos we want compute
    start = datetime.datetime(2023, 1, 1, 8, 0, 0)
    end = datetime.datetime(2025, 7, 15, 8, 0, 0)

    backtest_periods = (1, 7)

    # ------------------------------------
    # state all the alpha categories we want to compute
    alphas = [
        Alpha(
            category=Category.liquidity_1d,
            name="liq_amount_1",
            parent="liq_amount",
            freq="1d",
        ),
    ]

    # alphas = None

    # ------------------------------------
    # state all the necessary parameters
    by_group = True
    symbols = None
    just_metrics = True
    n_stratify = 5
    n_ic_layers = 3
    compound = True
    generate_pdf_charts = True

    groupby = GroupBy.amount_quarter_spot_3

    parent = Parent(category=Category.momentum_1d, name="mmt_alphawhole", freq="1d")

    parent = None

    data_sources = DataSources(
        factor=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
        kline=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
        group=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
    )

    # ------------------------------------
    # create the instance
    instance = SingleAnalyzer(
        data_sources=data_sources,
        groupby=groupby,
        n_ic_layers=n_ic_layers,
        alphas=alphas,
        parent=parent,
        symbols=symbols,
        by_group=by_group,
        compound=compound,
    )

    asyncio.run(
        instance.get_nvs_metrics(
            start=start,
            end=end,
            backtest_periods=backtest_periods,
            benchmark=Benchmark.whole,
            n_stratify=n_stratify,
            just_metrics=just_metrics,
            generate_pdf_charts=generate_pdf_charts,
        )
    )

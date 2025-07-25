import asyncio
import datetime
from json import load
import logging
import warnings
from pathlib import Path
from typing import Tuple, Union, List

import pandas as pd

from .performance import (
    information_coefficient_stats,
    rank_information_coefficient,
    information_coefficient,
)

from .stock_data import StockData
from .constants import *
from .utils import load_config, save_to_excel, send_file_to_windows, sublist

warnings.filterwarnings("ignore")

path_general = Path(load_config("path.yaml")["general"])
path_local_general = Path(load_config("path.yaml")["local_general"])
windows_server = load_config("server.yaml")["windows_server"]


class CategoryAnalyzer:
    def __init__(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        categories: List[Category],
        groupby: GroupBy,
        chunk_size: int,
        backtest_periods: Tuple,
        by_group: bool,
        symbols: Union[List[str], None] = None,
        sync: bool = False,
    ):

        self.start = start
        self.end = end
        self.categories = categories
        self.groupby = groupby
        self.chunk_size = chunk_size
        self.backtest_periods = backtest_periods
        self.by_group = by_group
        self.symbols = symbols
        self.sync = sync

    async def compute(self, data_sources: DataSources, aggregations: dict = dict()):

        for category_ in self.categories:

            path = (
                path_general
                / data_sources.factor.exchange.name
                / data_sources.factor.universe.name
                / "Alphas"
                / data_sources.factor.freq
                / category_.name
            )

            alpha_paths = list(path.glob("*.csv"))

            df_rankic_metrics, df_ic_metrics = self._compute(
                alpha_paths, data_sources, aggregations
            )

            # ------------------------------------
            # find the correct path, it should be results / groupby / kline_info / category

            path_remote = (
                path_general
                / data_sources.kline.exchange.name
                / data_sources.kline.universe.name
                / "Backtest"
                / self.groupby.name
                / category_.name
            )

            path_local = (
                path_local_general
                / data_sources.kline.exchange.name
                / data_sources.kline.universe.name
                / "Backtest"
                / self.groupby.name
                / category_.name
            )

            if not path_remote.exists():
                path_remote.mkdir(parents=True, exist_ok=True)

            # ------------------------------------
            # for the rankic, directly save
            save_to_excel(
                path=path_remote / f"summary_{category_.name}.xlsx",
                df=df_rankic_metrics,
                sheet_name="rankic_metrics",
                index=False,
            )

            # ------------------------------------
            # for the ic, directly save
            save_to_excel(
                path=path_remote / f"summary_{category_.name}.xlsx",
                df=df_ic_metrics,
                sheet_name="ic_metrics",
                index=False,
            )

            # ------------------------------------
            # sync if needed
            if self.sync:
                send_file_to_windows(
                    remote_path=path_remote / f"summary_{category_.name}.xlsx",
                    local_path=path_local / f"summary_{category_.name}.xlsx",
                    windows_username=windows_server["username"],
                    windows_ip=windows_server["ip"],
                    windows_password=windows_server["password"],
                )

    def _compute(
        self,
        alpha_paths: Union[List[str], List[Path]],
        data_sources: DataSources,
        aggregations: dict = dict(),
    ):

        # ------------------------------------
        # split the alpha paths into chunks to avoid overloading
        alphas_batch = sublist(alpha_paths, chunk_size=self.chunk_size)

        rankic_metrics_dfs = []
        ic_metrics_dfs = []

        for index, alphas_ in enumerate(alphas_batch):
            data = StockData(
                start_time=self.start,
                end_time=self.end,
                groupby=self.groupby,
                alpha_paths=alphas_,
                aggregations=aggregations,
                symbols=self.symbols,
                backtest_periods=self.backtest_periods,
                data_sources=data_sources,
            )

            # ------------------------------------
            # compute the rankic
            rankic_tensor = rank_information_coefficient(
                alphas=data.alphas,
                returns=data.returns,
                groups=data.groups,
                by_group=self.by_group,
            )

            # ------------------------------------
            # compute the ic
            ic_tensor = information_coefficient(
                alphas=data.alphas,
                returns=data.returns,
                groups=data.groups,
                by_group=self.by_group,
            )

            # ------------------------------------
            # compute the rankic metrics
            rankic_metrics_tensor = information_coefficient_stats(rankic_tensor)

            # ------------------------------------
            # compute the ic metrics
            ic_metrics_tensor = information_coefficient_stats(ic_tensor)

            # ------------------------------------
            # restore the rankic df
            df_rankic = data.make_dataframe(
                data=rankic_metrics_tensor,
                evaluation=Evaluation.ic_metrics,
                by_group=self.by_group,
            )

            df_ic = data.make_dataframe(
                data=ic_metrics_tensor,
                evaluation=Evaluation.ic_metrics,
                by_group=self.by_group,
            )

            rankic_metrics_dfs.append(df_rankic)
            ic_metrics_dfs.append(df_ic)

            logging.info(f"batch {index} is finished")

        # ------------------------------------
        # concatenate all the dfs for each batch
        df_rankic = pd.concat(rankic_metrics_dfs, ignore_index=True)
        df_ic = pd.concat(ic_metrics_dfs, ignore_index=True)

        return df_rankic, df_ic

    def get_summary(self, data_sources: DataSources):
        # --------------------------------------------
        # Loop 1: Loop over each category

        dfs_categories = []
        for category in self.categories:

            # --------------------------------------------
            # log the computing step
            logging.info(rf"-> now computing {category.name}")

            # --------------------------------------------
            # extract the frequency
            path_summary = (
                path_general
                / data_sources.kline.exchange.name
                / data_sources.kline.universe.name
                / "Backtest"
                / self.groupby.name
                / "K_24H_UTC0"
                / f"summary_{category.name}.xlsx"
            )

            if path_summary.exists():
                df_summary = pd.read_excel(
                    path_summary, sheet_name="ic_metrics", engine="openpyxl"
                )

                # --------------------------------------------
                # only keep the ic mean / positive_ratio / do the aggregation / do the renaming / add the category column
                df_summary = df_summary.loc[
                    (
                        df_summary["metrics"].isin(
                            ["MEAN", "POSITIVE_RATIO", "RISK_ADJUSTED_IC"]
                        )
                    )
                    & (df_summary["period"] == "7D")
                ]
                df_summary.drop(columns=["period"], inplace=True)

                # --------------------------------------------
                # add the category name
                df_summary["category"] = category.name
                df_summary = df_summary[
                    ["group", "category", "alpha", "metrics", "value"]
                ]
            else:
                continue

            # --------------------------------------------
            # append all the dfs
            dfs_categories.append(df_summary)

        # --------------------------------------------
        # concatenate all the summary dfs
        df_categories = pd.concat(dfs_categories, ignore_index=True)
        df_categories = df_categories.round(3)

        remote_save_path = (
            path_general
            / data_sources.kline.exchange.name
            / data_sources.kline.universe.name
            / "Backtest"
            / self.groupby.name
            / f"summary_alphas_statistics.xlsx"
        )

        local_save_path = (
            path_general
            / data_sources.kline.exchange.name
            / data_sources.kline.universe.name
            / "Backtest"
            / self.groupby.name
            / f"summary_alphas_statistics.xlsx"
        )

        save_to_excel(remote_save_path, df_categories, sheet_name="ic_summery")

        # --------------------------------------------
        # send the file from remote to local
        send_file_to_windows(
            remote_path=remote_save_path,
            local_path=local_save_path,
            windows_username=windows_server["username"],
            windows_ip=windows_server["ip"],
            windows_password=windows_server["password"],
        )


if __name__ == "__main__":
    # ------------------------------------
    # here we specify the start and end of the klines, NOT the alphas
    start = datetime.datetime(2023, 1, 1, 8, 0, 0)
    end = datetime.datetime(2025, 6, 14, 8, 0, 0)

    # ------------------------------------
    # state all the kline infos we want compute
    symbols = None

    groupby = GroupBy.amount_quarter_spot_3
    chunk_size = 5
    backtest_periods = (1, 3, 5, 7, 11, 13, 15)
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

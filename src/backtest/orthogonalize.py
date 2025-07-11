import datetime
import logging
import warnings
from pathlib import Path
from typing import Tuple, Union, List

import torch

import pandas as pd

from .performance import (
    information_coefficient_stats,
    rank_information_coefficient,
    information_coefficient,
    nanstd,
)

from .stock_data import StockData
from .constants import *
from .utils import load_config, save_to_excel, send_file_to_windows, sublist

warnings.filterwarnings("ignore")

path_general = Path(load_config("path.yaml")["general"])
path_local_general = Path(load_config("path.yaml")["local_general"])
windows_server = load_config("server.yaml")["windows_server"]

base_alphas = [
    Alpha(category=Category.liquidity_1d, alpha="liq_amount", freq="1d"),
    Alpha(category=Category.liquidity_1d, alpha="liq_trade", freq="1d"),
    Alpha(category=Category.liquidity_1h, alpha="liq_regress_24", freq="1h"),
    Alpha(category=Category.volatility_1d, alpha="vol_residual_20", freq="1d"),
]


class OrthogonalAnalyzer:
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

            alphas = list(path.glob("*.csv"))

            df_rankic_metrics, df_ic_metrics = self._compute(
                alphas, data_sources, aggregations
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
                path=path_remote / f"summary_orthogonalize_{category_.name}.xlsx",
                df=df_rankic_metrics,
                sheet_name="rankic_metrics",
                index=False,
            )

            # ------------------------------------
            # for the ic, directly save
            save_to_excel(
                path=path_remote / f"summary_orthogonalize_{category_.name}.xlsx",
                df=df_ic_metrics,
                sheet_name="ic_metrics",
                index=False,
            )

            # ------------------------------------
            # sync if needed
            if self.sync:
                send_file_to_windows(
                    remote_path=path_remote
                    / f"summary_orthogonalize_{category_.name}.xlsx",
                    local_path=path_local
                    / f"summary_orthogonalize_{category_.name}.xlsx",
                    windows_username=windows_server["username"],
                    windows_ip=windows_server["ip"],
                    windows_password=windows_server["password"],
                )

    def _compute(
        self,
        alphas: Union[List[str], List[Path]],
        data_sources: DataSources,
        aggregations: dict = dict(),
    ):

        # ------------------------------------
        # split the alpha paths into chunks to avoid overloading
        alphas_batch = sublist(alphas, chunk_size=self.chunk_size)

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

            base_alphas_ = [
                path_general
                / data_sources.factor.exchange.name
                / data_sources.factor.universe.name
                / alpha.path
                for alpha in base_alphas
            ]

            data_base = StockData(
                start_time=self.start,
                end_time=self.end,
                groupby=self.groupby,
                alpha_paths=base_alphas_,
                symbols=data._stock_ids,
                backtest_periods=self.backtest_periods,
                data_sources=data_sources,
            )

            data.alphas = self.orthogonalize(data.alphas, data_base.alphas)

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

    def orthogonalize(self, alphas: torch.Tensor, base_alphas: torch.Tensor):
        # alphas : (dates, alphas, stocks)
        # base_alphas : (dates, base_alphas, stocks)

        dates, num_alphas, stocks = alphas.shape
        dates_base, num_base_alphas, stocks_base = base_alphas.shape

        # Verify dimensions match
        assert dates == dates_base and stocks == stocks_base

        # Initialize output tensor
        orthogonalized_alphas = torch.zeros_like(alphas)

        # Process each date separately (cross-sectional regression)
        for date_idx in range(dates):
            # Get cross-sections for this date
            alpha_slice = alphas[date_idx]  # (num_alphas, stocks)
            base_slice = base_alphas[date_idx]  # (num_base_alphas, stocks)

            # Transpose for regression: (stocks, features)
            X = base_slice.T  # (stocks, num_base_alphas) - predictors
            Y = alpha_slice.T  # (stocks, num_alphas) - targets

            # Handle NaN values
            valid_mask = ~(torch.isnan(X).any(dim=1) | torch.isnan(Y).any(dim=1))

            if valid_mask.sum() < num_base_alphas:
                # Not enough valid data, raise problem
                raise

            # Extract valid data
            X_valid = X[valid_mask]  # (valid_stocks, num_base_alphas)
            Y_valid = Y[valid_mask]  # (valid_stocks, num_alphas)

            # Winsorization: clip extreme values to 1st and 99th percentiles
            X_winsorized = torch.zeros_like(X_valid)
            Y_winsorized = torch.zeros_like(Y_valid)

            for i in range(num_base_alphas):
                p3, p97 = torch.nanquantile(
                    X_valid[:, i], torch.tensor([0.03, 0.97], device=alphas.device)
                )
                X_winsorized[:, i] = torch.clamp(X_valid[:, i], p3, p97)

            for i in range(num_alphas):
                p3, p97 = torch.nanquantile(
                    Y_valid[:, i], torch.tensor([0.03, 0.97], device=alphas.device)
                )
                Y_winsorized[:, i] = torch.clamp(Y_valid[:, i], p3, p97)

            # Standardization: (x - mean) / std
            X_standardized = (
                X_winsorized - X_winsorized.nanmean(dim=0, keepdim=True)
            ) / (nanstd(X_winsorized, dim=0, keepdim=True))
            Y_standardized = (
                Y_winsorized - Y_winsorized.nanmean(dim=0, keepdim=True)
            ) / (nanstd(Y_winsorized, dim=0, keepdim=True))

            beta = (
                torch.linalg.pinv(X_standardized) @ Y_standardized
            )  # (num_base_alphas, num_alphas)

            # Compute fitted values and residuals for standardized data
            fitted_standardized = X_standardized @ beta  # (valid_stocks, num_alphas)
            residuals_standardized = (
                Y_standardized - fitted_standardized
            )  # (valid_stocks, num_alphas)

            # De-standardize residuals back to original scale
            Y_std = Y_winsorized.std(dim=0, keepdim=True) + 1e-8
            Y_mean = Y_winsorized.mean(dim=0, keepdim=True)
            residuals_destandardized = residuals_standardized * Y_std + Y_mean

            # Create full residuals tensor for all stocks
            residuals_full = torch.zeros_like(Y)  # (stocks, num_alphas)
            residuals_full[valid_mask] = residuals_destandardized
            residuals_full[~valid_mask] = Y[
                ~valid_mask
            ]  # Keep original for invalid stocks

            # Store result (transpose back)
            orthogonalized_alphas[date_idx] = residuals_full.T  # (num_alphas, stocks)

        return orthogonalized_alphas

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

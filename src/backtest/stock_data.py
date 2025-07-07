from itertools import chain
from typing import Tuple

import polars as pl
import pandas as pd

from .constants import *
from .utils import *

path_general = Path(load_config("path.yaml")["general"])


class StockData:

    def __init__(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        groupby: GroupBy,
        alpha_paths: List[Path],
        data_sources: DataSources,
        n_stratify: int = 5,
        aggregations: dict = dict(),
        symbols: Union[List, None] = None,
        backtest_periods: Tuple[int] = (1,),
        device: torch.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ) -> None:

        self._start_time = start_time
        self._end_time = end_time
        self.groupby = groupby
        self.n_stratify = n_stratify
        self.alpha_paths = alpha_paths
        self.symbols = symbols
        self.data_sources = data_sources
        self.aggregations = aggregations
        self.backtest_periods = backtest_periods
        self.device = device

        assert (
            self.backtest_periods[0] == 1
        ), "the first period must be 1, for the long short backtest purpose"

        self.kline_config = load_config("database.yaml")[
            data_sources.kline.exchange.name
        ]

        # operate
        self._check()
        self._format()
        self.alphas, self._dates, self._stock_ids, self._alpha_names = (
            self._get_alphas()
        )
        self.returns = self._get_returns()

        # if there is no groupby feed, assign the self.groups as None
        if self.groupby != GroupBy.no_group:
            self.groups = self._get_group()
        else:
            self.groups = None

        self.klines = self._get_kline()

        # mask alpha where there is no kline
        mask = torch.isnan(self.klines.unsqueeze(1).expand(self.alphas.shape))
        self.alphas[mask] = torch.nan

        logging.info(rf"everything has been put on {self.device}")

    def _check(self) -> None:
        """
        read the three kinds of data and do some checking
        A. alpha data
        B. kline data
        C. group data
        :return:
        """

        # read the alpha data
        # ------------------------------------

        offset = pd.Timedelta(self.data_sources.factor.freq) * (
            max(chain.from_iterable(self.aggregations.values()))
            if self.aggregations
            else 0
        )

        # Start with first file using pandas
        df_alphas = pd.read_csv(self.alpha_paths[0])

        # Sequentially outer merge remaining files horizontally using pandas
        for alpha_file in self.alpha_paths[1:]:
            df_to_join = pd.read_csv(alpha_file)
            df_alphas = pd.merge(
                df_alphas, df_to_join, on=["time", "symbol"], how="outer"
            )

        # Convert the final merged pandas DataFrame to polars DataFrame
        df_alphas = pl.from_pandas(df_alphas)
        df_alphas = df_alphas.sort(["time", "symbol"])

        # filter the time
        df_alphas = df_alphas.with_columns(
            pl.col("time")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .alias("time")
        )

        df_alphas = df_alphas.filter(
            pl.col("time").is_between(
                self._start_time - offset,
                self._end_time
                - datetime.timedelta(days=1) * max(self.backtest_periods),
                closed="both",  # inclusive of both dates
            )
        )

        # Calculate all requested aggregations
        for agg_method, windows in self.aggregations.items():
            for alpha in self.alpha_paths:
                for window in windows:

                    if agg_method == "STD":
                        df_alphas = df_alphas.with_columns(
                            pl.col(alpha.stem)
                            .rolling_std(window, min_periods=window)
                            .over("symbol")
                            .alias(f"{alpha.stem}_{agg_method}{window}")
                        )

                    elif agg_method == "MA":
                        df_alphas = df_alphas.with_columns(
                            pl.col(alpha.stem)
                            .rolling_mean(window, min_periods=window)
                            .over("symbol")
                            .alias(f"{alpha.stem}_{agg_method}{window}")
                        )

                    else:
                        raise ValueError("agg_method should be one of `MA` and `STD`")

        # rename the symbols, eventually all the symbols are in the format of `BTCUSDT`
        if (
            self.data_sources.factor.exchange == Exchange.Okx
            and self.data_sources.factor.universe == Universe.perp
        ):
            df_alphas = df_alphas.with_columns(
                pl.col("symbol").map_elements(lambda x: "".join(x.split("-")[:2]))
            )
        elif (
            self.data_sources.factor.exchange == Exchange.Okx
            and self.data_sources.factor.universe == Universe.spot
        ):
            df_alphas = df_alphas.with_columns(
                pl.col("symbol").map_elements(lambda x: x.replace("-", ""))
            )
        elif (
            self.data_sources.factor.exchange == Exchange.Binance
            and self.data_sources.factor.universe == Universe.perp
        ):
            df_alphas = df_alphas.with_columns(
                pl.col("symbol").map_elements(
                    lambda x: x.replace("1000000", "")
                    .replace("1000", "")
                    .replace("1MBABYDOGE", "BABYDOGE")
                )
            )
        elif (
            self.data_sources.factor.exchange == Exchange.Binance
            and self.data_sources.factor.universe == Universe.spot
        ):
            df_alphas = df_alphas.with_columns(
                pl.col("symbol").map_elements(
                    lambda x: x.replace("1000", "").replace("1MBABYDOGE", "BABYDOGE")
                )
            )
        else:
            pass

        self.df_alphas = df_alphas.to_pandas()

        # ------------------------------------
        # select the symbols
        if self.symbols is not None:
            self.df_alphas = self.df_alphas.loc[
                self.df_alphas["symbol"].isin(self.symbols)
            ]
            self.df_alphas.reset_index(drop=True, inplace=True)
        else:
            pass

        # ------------------------------------
        # convert them all to tensor
        self.df_alphas.set_index(["time", "symbol"], inplace=True)

        # 2) read the kline data
        # ------------------------------------
        # read the corresponding kline data according to the specified kline_info
        self._read_kline()

        self.df_returns = compute_forward_returns(
            factor=self.df_alphas,
            prices=self.df_kline,
            periods=self.backtest_periods,
            cumulative_returns=True,
        )

        self.df_alphas = self.df_alphas.stack(dropna=False).unstack(level=1)
        self.df_returns = self.df_returns.stack(dropna=False).unstack(level=1)

        self.df_alphas.index.set_names(["date", "alpha"], inplace=True)
        self.df_returns.index.set_names(["date", "period"], inplace=True)

        # 3) read the group data according to the kline_info and the specified groupby info
        # ------------------------------------
        # read the corresponding group data according to the specified kline_info
        if self.groupby != GroupBy.no_group:
            self._read_group()
        else:
            pass

        return

    def encode_groups(self, df: pd.DataFrame, group_column: str) -> pd.DataFrame:
        # Loop through the enum and replace group names with the corresponding numeric values
        if self.groupby.name.startswith("amount") and self.groupby.name.endswith("4"):
            for group_name, group_value in Amount4Group.__members__.items():
                df.loc[df[group_column] == group_name, group_column] = float(
                    group_value
                )
            return df
        elif self.groupby.name.startswith("amount") and self.groupby.name.endswith("3"):
            for group_name, group_value in Amount3Group.__members__.items():
                df.loc[df[group_column] == group_name, group_column] = float(
                    group_value
                )
            return df
        elif self.groupby.name.startswith("test") and self.groupby.name.endswith("3"):
            for group_name, group_value in Amount2Group.__members__.items():
                df.loc[df[group_column] == group_name, group_column] = float(
                    group_value
                )
            return df
        elif self.groupby.name.startswith("liquidity") and self.groupby.name.endswith(
            "3"
        ):
            for group_name, group_value in Liquidity3Group.__members__.items():
                df.loc[df[group_column] == group_name, group_column] = float(
                    group_value
                )
            return df
        else:
            raise ValueError("Under Development")

    def _format(self) -> None:
        return_date_index = self.df_returns.index.levels[0]

        # ------------------------------------
        # format the 3 dfs according to the self.df_returns
        self.df_alphas = self.df_alphas.reindex(return_date_index, level="date")
        self.df_kline = self.df_kline.reindex(return_date_index)

        # todo: delete this later
        # self.df_vol = self.df_vol.reindex(return_date_index)

        # ------------------------------------
        # here we use the common stocks in the three data since
        # 1. there maybe some new coins which are not in the self.df_alphas
        # 2. shortly lised and delisted coins may not be in the self.df_alphas and self.df_group

        if self.groupby != GroupBy.no_group:
            self.df_group = self.df_group.reindex(return_date_index)
            common_stocks = self.df_alphas.columns.intersection(
                self.df_returns.columns
            ).intersection(self.df_group.columns)
            common_stocks = sorted(common_stocks)

            # put `BTCUSDT` as the first one
            if "BTCUSDT" in common_stocks:
                common_stocks.remove("BTCUSDT")
                common_stocks.insert(0, "BTCUSDT")

            self.df_alphas = self.df_alphas[common_stocks]
            self.df_returns = self.df_returns[common_stocks]
            self.df_group = self.df_group[common_stocks]
            self.df_kline = self.df_kline[common_stocks]
        else:
            common_stocks = self.df_alphas.columns.intersection(self.df_returns.columns)
            common_stocks = sorted(common_stocks)
            self.df_alphas = self.df_alphas[common_stocks]
            self.df_returns = self.df_returns[common_stocks]
            self.df_kline = self.df_kline[common_stocks]

        logging.info("formatting is done")

        return

    def _get_kline(self, df_kline: Union[pd.DataFrame, None] = None) -> torch.Tensor:

        if df_kline is None:
            df_kline = self.df_kline.copy().astype(float)
        else:
            df_kline = df_kline.astype(float)

        # ------------------------------------
        # convert them all to tensor
        stock_ids = df_kline.columns
        values = df_kline.values
        values = values.reshape(-1, len(stock_ids))

        return torch.tensor(values, dtype=torch.float, device=self.device)

    def _get_alphas(self) -> Tuple[torch.Tensor, pd.Index, pd.Index, pd.Index]:
        df_alphas = self.df_alphas.copy()

        # ------------------------------------
        # convert them all to tensor
        dates = df_alphas.index.levels[0]

        alpha_names = df_alphas.index.levels[1]
        stock_ids = df_alphas.columns
        values = df_alphas.values
        values = values.reshape((-1, len(alpha_names), len(stock_ids)))

        data = torch.tensor(values, dtype=torch.float, device=self.device)

        # ------------------------------------
        # return the torch tensor and all the necessary index
        return (
            data,
            dates,
            stock_ids,
            alpha_names,
        )

    def _get_returns(self) -> torch.Tensor:
        df_returns = self.df_returns.copy()

        # ------------------------------------
        # convert them all to tensor
        periods = df_returns.index.levels[1]
        stock_ids = df_returns.columns
        values = df_returns.values
        values = values.reshape((-1, len(periods), len(stock_ids)))

        # ------------------------------------
        # return the torch tensor and all the necessary index
        return torch.tensor(values, dtype=torch.float, device=self.device)

    def _get_group(self) -> torch.Tensor:
        df_group = self.df_group.copy().astype(float)

        # ------------------------------------
        # convert them all to tensor
        stock_ids = df_group.columns
        values = df_group.values
        values = values.reshape(-1, len(stock_ids))

        return torch.tensor(values, dtype=torch.float, device=self.device)

    def _read_kline(self):
        """
        Returns None
        -------

        This method is used for reading the kline file and format the symbols according to the Binance symbol format
        """

        exchange = self.data_sources.kline.exchange.name
        universe = self.data_sources.kline.universe.name

        path_kline = (
            path_general
            / exchange
            / universe
            / "Klines"
            / f"{self.data_sources.kline.freq}.csv"
        )
        df_kline = pl.scan_csv(path_kline).collect()

        df_kline = df_kline.with_columns(
            pl.col("time")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .alias("time")
        )

        df_kline = df_kline.filter(
            pl.col("time").is_between(
                self._start_time - pd.Timedelta(self.data_sources.kline.freq),
                self._end_time,
                closed="both",  # inclusive of both dates
            )
        )

        # rename the symbols, eventually all the symbols are in the format of `BTCUSDT`
        if (
            self.data_sources.factor.exchange == Exchange.Okx
            and self.data_sources.factor.universe == Universe.perp
        ):
            df_kline = df_kline.with_columns(
                pl.col("symbol").map_elements(lambda x: "".join(x.split("-")[:2]))
            )
        elif (
            self.data_sources.factor.exchange == Exchange.Okx
            and self.data_sources.factor.universe == Universe.spot
        ):
            df_kline = df_kline.with_columns(
                pl.col("symbol").map_elements(lambda x: x.replace("-", ""))
            )
        elif (
            self.data_sources.factor.exchange == Exchange.Binance
            and self.data_sources.factor.universe == Universe.perp
        ):
            df_kline = df_kline.with_columns(
                pl.col("symbol").map_elements(
                    lambda x: x.replace("1000000", "")
                    .replace("1000", "")
                    .replace("1MBABYDOGE", "BABYDOGE")
                )
            )
        elif (
            self.data_sources.factor.exchange == Exchange.Binance
            and self.data_sources.factor.universe == Universe.spot
        ):
            df_kline = df_kline.with_columns(
                pl.col("symbol").map_elements(
                    lambda x: x.replace("1000", "").replace("1MBABYDOGE", "BABYDOGE")
                )
            )
        else:
            pass

        self.df_kline = df_kline.to_pandas()

        # ------------------------------------
        # filter out the columns we need
        self.df_kline = self.df_kline[["time", "symbol", "close"]]
        self.df_kline["time"] = self.df_kline["time"] + pd.Timedelta(
            self.data_sources.kline.freq
        )

        self.df_kline.set_index(["time", "symbol"], inplace=True)
        self.df_kline = self.df_kline.stack(dropna=False).unstack(level=1)
        self.df_kline = self.df_kline.droplevel(1)

    def _read_group(self):
        """
        Returns None
        -------

        This method is used for reading the kline file and format the symbols according to the Binance symbol format
        """
        exchange = self.data_sources.group.exchange.name
        universe = self.data_sources.group.universe.name

        path_group = (
            path_general / exchange / universe / "Groups" / f"{self.groupby.name}.csv"
        )

        df_group = pl.scan_csv(path_group).collect()
        df_group = df_group.rename({"class": "group"})
        df_group = df_group.drop("id")
        df_group = df_group.with_columns(
            pl.col("time")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .alias("time")
        )

        # rename the symbols, eventually all the symbols are in the format of `BTCUSDT`
        if (
            self.data_sources.factor.exchange == Exchange.Okx
            and self.data_sources.factor.universe == Universe.perp
        ):
            df_group = df_group.with_columns(
                pl.col("symbol").map_elements(lambda x: "".join(x.split("-")[:2]))
            )
        elif (
            self.data_sources.factor.exchange == Exchange.Okx
            and self.data_sources.factor.universe == Universe.spot
        ):
            df_group = df_group.with_columns(
                pl.col("symbol").map_elements(lambda x: x.replace("-", ""))
            )
        elif (
            self.data_sources.factor.exchange == Exchange.Binance
            and self.data_sources.factor.universe == Universe.perp
        ):
            df_group = df_group.with_columns(
                pl.col("symbol").map_elements(
                    lambda x: x.replace("1000000", "")
                    .replace("1000", "")
                    .replace("1MBABYDOGE", "BABYDOGE")
                )
            )
        elif (
            self.data_sources.factor.exchange == Exchange.Binance
            and self.data_sources.factor.universe == Universe.spot
        ):
            df_group = df_group.with_columns(
                pl.col("symbol").map_elements(
                    lambda x: x.replace("1000", "").replace("1MBABYDOGE", "BABYDOGE")
                )
            )
        else:
            pass

        self.df_group = df_group.to_pandas()
        self.df_group["time"] = self.df_group["time"] + pd.Timedelta(
            self.data_sources.group.freq
        )

        self.df_group = self.encode_groups(self.df_group, group_column="group")

        self.df_group.set_index(["time", "symbol"], inplace=True)
        self.df_group = self.df_group.stack(dropna=False).unstack(level=1)
        self.df_group = self.df_group.droplevel(1)

    @property
    def n_stocks(self) -> int:
        return len(self._stock_ids)

    @property
    def n_dates(self) -> int:
        return len(self._dates)

    @property
    def n_groups(self) -> int:
        if self.groupby.name.startswith("Amount") and self.groupby.name.endswith("4"):
            return len(list(Amount4Group))
        elif self.groupby.name.startswith("Amount") and self.groupby.name.endswith("3"):
            return len(list(Amount3Group))
        else:
            raise ValueError("under developing")

    @property
    def n_alphas(self) -> int:
        return len(self._alpha_names)

    @property
    def n_modes(self) -> int:
        return len(self.BacktestModes)

    @property
    def n_layers(self) -> int:
        return self.n_stratify

    @property
    def periods(self) -> List[str]:
        freq = self.data_sources.kline.freq[-1].upper()

        periods = list(self.backtest_periods)
        periods = [rf"{period}{freq}" for period in periods]
        return periods

    @property
    def layers(self) -> List[str]:
        periods = [rf"layer_{layer}" for layer in range(1, 1 + self.n_stratify)]
        return periods

    @property
    def groups_names(self) -> List[str]:
        if self.groupby.name.startswith("amount") and self.groupby.name.endswith("4"):
            return list(Amount4Group.__members__.keys())
        elif self.groupby.name.startswith("amount") and self.groupby.name.endswith("3"):
            return list(Amount3Group.__members__.keys())
        elif self.groupby.name.startswith("liquidity") and self.groupby.name.endswith(
            "3"
        ):
            return list(Liquidity3Group.__members__.keys())
        else:
            raise ValueError("under developing")

    @property
    def metrics_ic(self) -> List[str]:
        return list(Metrics_IC.__members__.keys())

    @property
    def metrics_stratify(self) -> List[str]:
        return list(Metrics_Stratify.__members__.keys())

    @property
    def BacktestModes(self) -> List[str]:
        return list(BacktestModes.__members__.keys())

    @property
    def Stratifications(self) -> List[str]:
        return list(BacktestModes.__members__.keys())

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        evaluation: Evaluation,
        by_group: bool = True,
    ) -> pd.DataFrame:
        """
        Parameters:
        - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
        a list of tensors of size `(n_days, n_stocks)`
        - `type`: a str among ['ic', 'nv', 'metrics', 'maybe others']
        - `groupby`: specify whether the tensor is grouped
        """

        if evaluation == Evaluation.ic_metrics:

            if by_group:
                n_groups, n_alphas, n_periods, n_metrics, _ = data.shape

                index = pd.MultiIndex.from_product(
                    [
                        self.groups_names,
                        self._alpha_names,
                        self.periods,
                        self.metrics_ic,
                    ]
                )
                data = data.reshape(-1, 1)

                df = pd.DataFrame(
                    data.detach().cpu().numpy(), index=index, columns=["value"]
                ).reset_index()
                df.rename(
                    columns={
                        "level_0": "group",
                        "level_2": "period",
                        "level_3": "metrics",
                    },
                    inplace=True,
                )

                logging.info(rf"-> {evaluation.name} has been generated")

                return df

            else:
                n_alphas, n_periods, n_metrics, _ = data.shape

                index = pd.MultiIndex.from_product(
                    [self._alpha_names, self.periods, self.metrics_ic]
                )
                data = data.reshape(-1, 1)

                df = pd.DataFrame(
                    data.detach().cpu().numpy(), index=index, columns=["value"]
                ).reset_index()
                df.rename(
                    columns={"level_1": "period", "level_2": "metrics"}, inplace=True
                )

                logging.info(rf"-> {evaluation.name} has been generated")

                return df

        elif evaluation == Evaluation.ics:

            if by_group:
                n_dates, n_groups, n_alphas, n_periods, _ = data.shape

                index = pd.MultiIndex.from_product(
                    [self._dates, self.groups_names, self._alpha_names, self.periods]
                )
                data = data.reshape(-1, 1)

                df = pd.DataFrame(
                    data.detach().cpu().numpy(), index=index, columns=["value"]
                ).reset_index()
                df.rename(
                    columns={"level_1": "group", "level_3": "period"}, inplace=True
                )

                logging.info(rf"-> {evaluation.name} has been generated")

                return df

            else:
                n_dates, n_alphas, n_periods, _ = data.shape

                index = pd.MultiIndex.from_product(
                    [self._dates, self._alpha_names, self.periods]
                )
                data = data.reshape(-1, 1)

                df = pd.DataFrame(
                    data.detach().cpu().numpy(), index=index, columns=["value"]
                ).reset_index()
                df.rename(columns={"level_2": "periods"}, inplace=True)

                logging.info(rf"-> {evaluation.name} has been generated")

                return df

        elif evaluation in [Evaluation.nvs, Evaluation.drawdowns, Evaluation.returns]:

            if by_group:
                n_dates, n_alphas, n_groups, n_modes, n_periods, _ = data.shape

                index = pd.MultiIndex.from_product(
                    [
                        self._dates,
                        self._alpha_names,
                        self.groups_names,
                        self.BacktestModes,
                        self.periods,
                    ]
                )
                data = data.reshape(-1, 1)

                df = pd.DataFrame(
                    data.detach().cpu().numpy(), index=index, columns=["value"]
                ).reset_index()
                df.rename(
                    columns={
                        "level_2": "group",
                        "level_3": "mode",
                        "level_4": "period",
                    },
                    inplace=True,
                )

                logging.info(rf"-> {evaluation.name} has been generated")

                return df

            else:
                n_dates, n_alphas, n_modes, n_periods, _ = data.shape

                index = pd.MultiIndex.from_product(
                    [self._dates, self._alpha_names, self.BacktestModes, self.periods]
                )
                data = data.reshape(-1, 1)

                df = pd.DataFrame(
                    data.detach().cpu().numpy(), index=index, columns=["value"]
                ).reset_index()
                df.rename(
                    columns={"level_2": "mode", "level_3": "period"}, inplace=True
                )

                logging.info(rf"-> {evaluation.name} has been generated")

                return df

        elif evaluation == Evaluation.turnover:

            if by_group:
                n_alphas, n_groups, n_modes, n_periods, _ = data.shape

                index = pd.MultiIndex.from_product(
                    [
                        self._alpha_names,
                        self.groups_names,
                        self.BacktestModes,
                        self.periods,
                    ]
                )
                data = data.reshape(-1, 1)

                df = pd.DataFrame(
                    data.detach().cpu().numpy(), index=index, columns=["value"]
                ).reset_index()
                df.rename(
                    columns={
                        "level_1": "group",
                        "level_2": "mode",
                        "level_3": "period",
                    },
                    inplace=True,
                )

                logging.info(rf"-> {evaluation.name} has been generated")

                return df

            else:
                n_alphas, n_modes, n_periods, _ = data.shape

                index = pd.MultiIndex.from_product(
                    [self._alpha_names, self.BacktestModes, self.periods]
                )
                data = data.reshape(-1, 1)

                df = pd.DataFrame(
                    data.detach().cpu().numpy(), index=index, columns=["value"]
                ).reset_index()
                df.rename(
                    columns={"level_1": "mode", "level_2": "period"}, inplace=True
                )

                logging.info(rf"-> {evaluation.name} has been generated")

                return df

    def make_dataframe_stratification(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        evaluation: Evaluation,
        by_group: bool = True,
    ) -> pd.DataFrame:
        """
        Parameters:
        - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
        a list of tensors of size `(n_days, n_stocks)`
        - `type`: a str among ['ic', 'nv', 'metrics', 'maybe others']
        - `groupby`: specify whether the tensor is grouped
        """

        if evaluation in [Evaluation.nvs, Evaluation.returns]:

            if by_group:
                n_dates, n_alphas, n_groups, n_layers, n_periods, _ = data.shape

                index = pd.MultiIndex.from_product(
                    [
                        self._dates,
                        self._alpha_names,
                        self.groups_names,
                        self.layers,
                        self.periods,
                    ]
                )
                data = data.reshape(-1, 1)

                df = pd.DataFrame(
                    data.detach().cpu().numpy(), index=index, columns=["value"]
                ).reset_index()
                df.rename(
                    columns={
                        "level_2": "group",
                        "level_3": "layer",
                        "level_4": "period",
                    },
                    inplace=True,
                )

                logging.info(rf"-> stratify {evaluation.name} has been generated")

                return df

            else:
                n_dates, n_alphas, n_layers, n_periods, _ = data.shape

                index = pd.MultiIndex.from_product(
                    [self._dates, self._alpha_names, self.layers, self.periods]
                )
                data = data.reshape(-1, 1)

                df = pd.DataFrame(
                    data.detach().cpu().numpy(), index=index, columns=["value"]
                ).reset_index()
                df.rename(
                    columns={"level_2": "layer", "level_3": "period"}, inplace=True
                )

                logging.info(rf"-> stratify {evaluation.name} has been generated")

                return df

        elif evaluation == Evaluation.ret_metrics:

            if by_group:
                n_alphas, n_groups, n_layers, n_periods, n_metrics, _ = data.shape

                index = pd.MultiIndex.from_product(
                    [
                        self._alpha_names,
                        self.groups_names,
                        self.layers,
                        self.periods,
                        self.metrics_stratify,
                    ]
                )
                data = data.reshape(-1, 1)

                df = pd.DataFrame(
                    data.detach().cpu().numpy(), index=index, columns=["value"]
                ).reset_index()
                df.rename(
                    columns={
                        "level_1": "group",
                        "level_2": "layer",
                        "level_3": "period",
                        "level_4": "metrics",
                    },
                    inplace=True,
                )

                logging.info(rf"-> stratify {evaluation.name} has been generated")

                return df

            else:
                n_alphas, n_layers, n_periods, n_metrics, _ = data.shape

                index = pd.MultiIndex.from_product(
                    [
                        self._alpha_names,
                        self.layers,
                        self.periods,
                        self.metrics_stratify,
                    ]
                )
                data = data.reshape(-1, 1)

                df = pd.DataFrame(
                    data.detach().cpu().numpy(), index=index, columns=["value"]
                ).reset_index()
                df.rename(
                    columns={
                        "level_1": "layer",
                        "level_2": "period",
                        "level_3": "metrics",
                    },
                    inplace=True,
                )

                logging.info(rf"-> stratify {evaluation.name} has been generated")

                return df

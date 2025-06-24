import asyncio
import warnings

from scipy.stats import boltzmann

import quantstats as qs
from database import Reader
from .performance import *
from .stock_data import StockData
from .constants import *
from .utils import *

warnings.filterwarnings("ignore")

path_general = Path(load_config("path.yaml")["general"])
path_local_general = Path(load_config("path.yaml")["local_general"])
windows_server = load_config("server.yaml")["windows_server"]


class SingleAnalyzer:
    def __init__(
        self,
        groupby: GroupBy,
        alphas: List[Alpha],
        by_group: bool,
        equal_weight: bool,
        symbols: Union[List, None] = None,
        sync: bool = False,
    ):

        self.groupby = groupby
        self.symbols = symbols
        self.alphas = alphas
        self.by_group = by_group
        self.equal_weight = equal_weight
        self.sync = sync

    async def get_nvs_metrics(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        backtest_periods: Tuple,
        data_sources: DataSources,
        benchmark: Benchmark = Benchmark.btc,
        just_metrics: bool = True,
        n_stratify: int = 5,
    ):

        # ------------------------------------
        # make the data
        for alpha in self.alphas:

            alphas = [
                path_general
                / data_sources.factor.exchange.name
                / data_sources.factor.universe.name
                / alpha.path
            ]

            data = StockData(
                start_time=start,
                end_time=end,
                groupby=self.groupby,
                n_stratify=n_stratify,
                alphas=alphas,
                aggregations=alpha.aggregations,
                symbols=self.symbols,
                backtest_periods=backtest_periods,
                data_sources=data_sources,
            )

            # ------------------------------------
            # compute the ic
            ic_tensor = rank_information_coefficient(
                alphas=data.alphas,
                returns=data.returns,
                groups=data.groups,
                by_group=self.by_group,
            )

            # ------------------------------------
            # compute the metrics
            metrics_tensor = information_coefficient_stats(ic_tensor)

            # ------------------------------------
            # get the long short returns
            returns_tensor, turnover_tensor = long_short_backtest(
                alphas=data.alphas,
                returns=data.returns,
                metrics=metrics_tensor,
                backtest_periods=backtest_periods,
                groups=data.groups,
                by_group=self.by_group,
                equal_weight=self.equal_weight,
            )

            # ------------------------------------
            # get the stratified returns
            stratified_returns_tensor, _ = stratified_backtest(
                alphas=data.alphas,
                returns=data.returns,
                metrics=metrics_tensor,
                backtest_periods=backtest_periods,
                groups=data.groups,
                by_group=self.by_group,
                n_stratify=n_stratify,
            )

            # ------------------------------------
            # get the long short net values / stratified net values
            nvs_longshort_tensor = return2netvalue(returns_tensor)
            dd_longshort_tensor = return2drawdown(returns_tensor)
            cumic_longshort_tensor = ic2cumsum(ic_tensor)

            nvs_stratify_tensor = return2netvalue(stratified_returns_tensor)

            # ------------------------------------
            # get the stratified dfs
            df_stratify_nvs = data.make_dataframe_stratification(
                data=nvs_stratify_tensor,
                evaluation=Evaluation.nvs,
                by_group=self.by_group,
            )

            # ------------------------------------
            # get the long short dfs
            df_cumics = data.make_dataframe(
                data=cumic_longshort_tensor,
                evaluation=Evaluation.ics,
                by_group=self.by_group,
            )
            df_longshort_nvs = data.make_dataframe(
                data=nvs_longshort_tensor,
                evaluation=Evaluation.nvs,
                by_group=self.by_group,
            )
            df_longshort_drawdowns = data.make_dataframe(
                data=dd_longshort_tensor,
                evaluation=Evaluation.drawdowns,
                by_group=self.by_group,
            )
            df_longshort_returns = data.make_dataframe(
                data=returns_tensor,
                evaluation=Evaluation.returns,
                by_group=self.by_group,
            )
            df_longshort_turnover = data.make_dataframe(
                data=turnover_tensor,
                evaluation=Evaluation.turnover,
                by_group=self.by_group,
            )

            await self.save_alpha(
                data=data,
                category=alpha.category,
                df_longshort_nvs=df_longshort_nvs,
                df_stratify_nvs=df_stratify_nvs,
                df_cumics=df_cumics,
                benchmark=benchmark,
                df_longshort_returns=df_longshort_returns,
                df_longshort_turnover=df_longshort_turnover,
                df_longshort_drawdowns=df_longshort_drawdowns,
                just_metrics=just_metrics,
                data_sources=data_sources,
            )

    async def save_alpha(
        self,
        data: StockData,
        category: Category,
        df_longshort_returns: pd.DataFrame,
        df_longshort_nvs: pd.DataFrame,
        df_stratify_nvs: pd.DataFrame,
        df_cumics: pd.DataFrame,
        df_longshort_turnover: pd.DataFrame,
        df_longshort_drawdowns: pd.DataFrame,
        data_sources: DataSources,
        just_metrics: bool = True,
        benchmark: Benchmark = Benchmark.btc,
    ):

        # set the pivot columns according to the groupby type
        if self.groupby != GroupBy.no_group:
            columns_pivot = ["group", "mode", "period"]
            prefix = ""
        else:
            columns_pivot = ["mode", "period"]
            df_longshort_turnover["group"] = "ALPHA"
            prefix = "ALPHA-"

        # get the benchmark 24H-UTC8 return info / make it a pd.Series
        async with Reader(data.kline_config) as reader:

            query = f"""
            SELECT time, ret
            FROM {benchmark.name}_{data.data_sources.kline.freq}
            """

            result = await reader.execute_query(
                query,
                (),
            )

        columns = ["time", "ret"]

        benchmark_return = pd.DataFrame(result, columns=columns)
        benchmark_return.set_index("time", inplace=True)

        # ------------------------------------
        # get the benchmark net value for the specified category / get the true start / true end
        real_start = data._dates.min()
        real_end = data._dates.max()

        # ------------------------------------
        # get the benchmark net value ranging from the real_start to real_end
        benchmark_nv = benchmark_return.loc[
            (benchmark_return.index >= real_start)
            & (benchmark_return.index <= real_end)
        ].copy()
        benchmark_nv = (benchmark_nv + 1).cumprod()
        benchmark_nv.iloc[0] = 1
        benchmark_nv.reset_index(inplace=True)

        # ------------------------------------
        # loop over each alpha and save them one by one
        for _alpha_name in data._alpha_names:

            # ------------------------------------
            # find the correct path, it should be results / groupby / kline_info / category
            remote_dir = (
                path_general
                / data_sources.kline.exchange.name
                / data_sources.kline.universe.name
                / "Backtest"
                / self.groupby.name
                / category.name
            )

            local_dir = (
                path_local_general
                / data_sources.kline.exchange.name
                / data_sources.kline.universe.name
                / "Backtest"
                / self.groupby.name
                / category.name
            )

            if not remote_dir.exists():
                remote_dir.mkdir(parents=True, exist_ok=True)
            # ------------------------------------
            # get the metrics from the df_returns
            # todo: modify later
            df_temp_returns = df_longshort_returns.loc[
                df_longshort_returns["alpha"] == _alpha_name
            ].copy()
            df_temp_returns.drop(columns=["alpha"], inplace=True)

            df_returns = df_temp_returns.copy()

            df_temp_returns = df_temp_returns.pivot_table(
                index="date", columns=columns_pivot, values="value", fill_value=None
            )

            df_temp_returns.columns = df_temp_returns.columns.map(
                lambda x: prefix + "-".join(x)
            )

            df_metrics = qs.reports.metrics(
                returns=df_temp_returns,
                benchmark=benchmark_return,
                rf=0.0,
                display=False,
                mode="basic",
                sep=False,
                internal="True",
                compounded=True,
                periods_per_year=365,
                match_dates=False,
            )

            # ------------------------------------
            # duplicate the benchmark column and assign the corresponding names
            cols_benchmark = [
                "BENCHMARK-" + mode_ + "-" + period_
                for mode_ in data.BacktestModes
                for period_ in data.periods
            ]
            for col_name in cols_benchmark:
                df_metrics[col_name] = df_metrics["Benchmark"]
            df_metrics.drop(columns=["Benchmark"], inplace=True)

            df_metrics = df_metrics.T
            df_metrics = df_metrics.reset_index()
            df_metrics[["group", "mode", "period"]] = df_metrics["index"].str.split(
                "-", expand=True
            )
            df_metrics.drop(columns=["index"], inplace=True)
            df_metrics.set_index(["group", "mode", "period"], inplace=True)
            df_metrics = (
                df_metrics.stack()
                .reset_index()
                .rename(columns={"level_3": "metric", 0: "value"})
            )

            # ------------------------------------
            # add the corresponding turnover rate
            df_temp_turnover = df_longshort_turnover.loc[
                df_longshort_turnover["alpha"] == _alpha_name
            ].copy()
            df_temp_turnover.drop(columns=["alpha"], inplace=True)
            df_drawdown = df_longshort_drawdowns.loc[
                df_longshort_drawdowns["alpha"] == _alpha_name
            ].copy()
            df_drawdown.drop(columns=["alpha"], inplace=True)
            df_temp_turnover["metric"] = "Turnover"
            df_temp_turnover["value"] = df_temp_turnover["value"].round(2)
            df_temp_turnover["value"] = df_temp_turnover["value"].astype(str)
            df_metrics = pd.concat([df_metrics, df_temp_turnover], ignore_index=True)

            # ------------------------------------
            # select the corresponding alpha
            df_longshort_nv = df_longshort_nvs.loc[
                df_longshort_nvs["alpha"] == _alpha_name
            ].copy()
            df_longshort_nv.drop(columns=["alpha"], inplace=True)

            df_stratify_nv = df_stratify_nvs.loc[
                df_stratify_nvs["alpha"] == _alpha_name
            ].copy()
            df_stratify_nv.drop(columns=["alpha"], inplace=True)

            df_cumic = df_cumics.loc[df_cumics["alpha"] == _alpha_name].copy()
            df_cumic.drop(columns=["alpha"], inplace=True)

            save_to_excel(
                path=remote_dir / rf"{_alpha_name}.xlsx",
                df=df_metrics,
                sheet_name="Metrics",
                index=False,
            )

            if not just_metrics:

                save_to_excel(
                    path=remote_dir / rf"{_alpha_name}.xlsx",
                    df=df_longshort_nv,
                    sheet_name="LongShortNV",
                    index=False,
                )

                save_to_excel(
                    path=remote_dir / rf"{_alpha_name}.xlsx",
                    df=df_returns,
                    sheet_name="LongShortReturns",
                    index=False,
                )

                save_to_excel(
                    path=remote_dir / rf"{_alpha_name}.xlsx",
                    df=df_stratify_nv,
                    sheet_name="StratifyNV",
                    index=False,
                )

                save_to_excel(
                    path=remote_dir / rf"{_alpha_name}.xlsx",
                    df=df_drawdown,
                    sheet_name="Drawdowns",
                    index=False,
                )

                save_to_excel(
                    path=remote_dir / rf"{_alpha_name}.xlsx",
                    df=df_cumic,
                    sheet_name="CumIC",
                    index=False,
                )

                save_to_excel(
                    path=remote_dir / rf"{_alpha_name}.xlsx",
                    df=benchmark_nv,
                    sheet_name="BenchmarkNV",
                    index=False,
                )

            if self.sync:

                # --------------------------------------------
                # send the file from remote to local
                send_file_to_windows(
                    remote_path=remote_dir / rf"{_alpha_name}.xlsx",
                    local_path=local_dir / rf"{_alpha_name}.xlsx",
                    windows_username=windows_server["username"],
                    windows_ip=windows_server["ip"],
                    windows_password=windows_server["password"],
                )

                logging.info(f"{_alpha_name} backtesting has been saved")

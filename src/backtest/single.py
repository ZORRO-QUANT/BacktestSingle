import subprocess
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import quantstats as qs
from database import Reader

from .constants import *
from .latex import *
from .performance import *
from .stock_data import StockData
from .utils import *

# Set the style for better-looking plots
try:
    plt.style.use("seaborn-v0_8")
except OSError:
    plt.style.use("seaborn")
sns.set_palette("husl")

warnings.filterwarnings("ignore")


path_general = Path(load_config("path.yaml")["general"])

path_local_general = Path(load_config("path.yaml")["local_general"])

windows_server = load_config("server.yaml")["windows_server"]


class SingleAnalyzer:

    def __init__(
        self,
        data_sources: DataSources,
        groupby: GroupBy,
        by_group: bool,
        n_ic_layers: int,
        parent: Optional[Parent] = None,
        alphas: Optional[List[Alpha]] = None,
        symbols: Union[List, None] = None,
        compound: bool = False,
        sync: bool = False,
    ):

        self.data_sources = data_sources
        self.groupby = groupby
        self.n_ic_layers = n_ic_layers
        self.symbols = symbols
        self.alphas = alphas

        self.by_group = by_group
        self.parent = parent
        self.compound = compound

        self.sync = sync

        assert (parent is None) != (
            alphas is None
        ), "Exactly one of parent or alphas must be should provided"

        # ------------------------------------

        # re-assign the alphas if self.alphas is None

        if self.alphas is None:

            self.alphas = self._get_children()

    def _get_children(self):

        # ------------------------------------

        # get all the children alphas which belong to the parent

        path_folder = (
            path_general
            / self.data_sources.factor.exchange.name
            / self.data_sources.factor.universe.name
            / "Alphas"
            / self.parent.freq
            / self.parent.category.name
        )

        alpha_names = [
            full_name.stem.split(">")[0]
            for full_name in list(path_folder.glob(f"*>{self.parent.name}.csv"))
        ]

        # ------------------------------------

        # create all the alphas from the alpha_names

        alphas = [
            Alpha(
                category=self.parent.category,
                name=alpha,
                freq=self.parent.freq,
                parent=self.parent.name,
            )
            for alpha in alpha_names
        ]

        return alphas

    async def get_nvs_metrics(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        backtest_periods: Tuple,
        benchmark: Benchmark = Benchmark.whole,
        just_metrics: bool = True,
        n_stratify: int = 5,
        generate_pdf_charts: bool = False,
    ):

        # ------------------------------------

        # make the data

        for alpha in self.alphas:

            alpha_paths = [
                path_general
                / self.data_sources.factor.exchange.name
                / self.data_sources.factor.universe.name
                / alpha.path
            ]

            data = StockData(
                start_time=start,
                end_time=end,
                groupby=self.groupby,
                n_stratify=n_stratify,
                n_ic_layers=self.n_ic_layers,
                alpha_paths=alpha_paths,
                aggregations=alpha.aggregations,
                symbols=self.symbols,
                backtest_periods=backtest_periods,
                data_sources=self.data_sources,
            )

            # ------------------------------------
            # compute the ic
            ic_tensor = rank_information_coefficient(
                alphas=data.alphas,
                returns=data.returns,
                groups=data.groups,
                n_stratify=self.n_ic_layers,
                by_group=self.by_group,
            )

            # ------------------------------------
            # compute the metrics
            metrics_tensor = information_coefficient_stats(ic_tensor)
            metrics_tensor_whole = metrics_tensor[..., -1, :, :, :, :]

            # ------------------------------------
            # get the long short returns
            returns_tensor, turnover_tensor = long_short_backtest(
                alphas=data.alphas,
                returns=data.returns,
                metrics=metrics_tensor_whole,
                backtest_periods=backtest_periods,
                groups=data.groups,
                by_group=self.by_group,
            )

            # ------------------------------------
            # get the stratified returns
            returns_stratified_tensor, stratified_turnover_tensor = stratified_backtest(
                alphas=data.alphas,
                returns=data.returns,
                metrics=metrics_tensor_whole,
                backtest_periods=backtest_periods,
                groups=data.groups,
                by_group=self.by_group,
                n_stratify=n_stratify,
            )

            # ------------------------------------
            # get the long short net values / drawdown tensors / cumic tensors
            nvs_longshort_tensor = return2netvalue(
                returns_tensor, compound=self.compound
            )

            cumic_longshort_tensor = ic2cumsum(ic_tensor)

            # ------------------------------------
            # get the stratify net values
            nvs_stratify_tensor = return2netvalue(
                returns_stratified_tensor, compound=self.compound
            )

            # ------------------------------------
            # get the stratified dfs
            df_stratify_nvs = data.make_dataframe_stratification(
                data=nvs_stratify_tensor,
                evaluation=Evaluation.nvs,
                by_group=self.by_group,
            )

            df_stratify_returns = data.make_dataframe_stratification(
                data=returns_stratified_tensor,
                evaluation=Evaluation.returns,
                by_group=self.by_group,
            )

            df_stratify_turnovers = data.make_dataframe_stratification(
                data=stratified_turnover_tensor,
                evaluation=Evaluation.turnover,
                by_group=self.by_group,
            )

            # ------------------------------------
            # restore the raw alphas for distribution plotting
            df_alphas = data.make_dataframe(
                data=data.alphas,
                evaluation=Evaluation.alphas,
                by_group=self.by_group,
            )

            # ------------------------------------
            # restore the rankic df
            df_ic_metrics = data.make_dataframe(
                data=metrics_tensor,
                evaluation=Evaluation.ic_metrics,
                by_group=self.by_group,
            )

            # ------------------------------------
            # get the long short dfs
            df_ics = data.make_dataframe(
                data=ic_tensor,
                evaluation=Evaluation.ics,
                by_group=self.by_group,
            )

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

            # ------------------------------------
            # save all the long short dfs
            await self.save(
                data=data,
                category=alpha.category,
                df_alphas=df_alphas,
                df_ics=df_ics,
                df_cumics=df_cumics,
                df_ic_metrics=df_ic_metrics,
                df_longshort_returns=df_longshort_returns,
                df_longshort_nvs=df_longshort_nvs,
                df_longshort_turnover=df_longshort_turnover,
                df_stratify_returns=df_stratify_returns,
                df_stratify_nvs=df_stratify_nvs,
                df_stratify_turnovers=df_stratify_turnovers,
                data_sources=self.data_sources,
                just_metrics=just_metrics,
                benchmark=benchmark,
                generate_pdf_charts=generate_pdf_charts,
            )

    async def save(
        self,
        data: StockData,
        category: Category,
        df_alphas: pd.DataFrame,
        df_ics: pd.DataFrame,
        df_cumics: pd.DataFrame,
        df_ic_metrics: pd.DataFrame,
        df_longshort_returns: pd.DataFrame,
        df_longshort_nvs: pd.DataFrame,
        df_longshort_turnover: pd.DataFrame,
        df_stratify_returns: pd.DataFrame,
        df_stratify_nvs: pd.DataFrame,
        df_stratify_turnovers: pd.DataFrame,
        data_sources: DataSources,
        just_metrics: bool = True,
        benchmark: Benchmark = Benchmark.btc,
        generate_pdf_charts: bool = False,
    ):

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

        # loop over each alpha and save them one by one
        for _alpha_name in data._alpha_names:

            # ------------------------------------
            # 1) get all the stuff necessary for the longshort nvs
            # set the pivot columns according to the groupby type
            if self.groupby != GroupBy.no_group:
                columns_pivot = ["group", "mode", "period"]
                prefix = ""
            else:
                columns_pivot = ["mode", "period"]
                df_longshort_turnover["group"] = "WHOLE"
                prefix = "WHOLE-"

            # ------------------------------------
            # find the correct path, it should be results / groupby / kline_info / category
            remote_dir = (
                path_general
                / data_sources.kline.exchange.name
                / data_sources.kline.universe.name
                / "Backtest"
                / self.groupby.name
                / category.name
                / rf"{_alpha_name}>{data.parent_map[_alpha_name]}"
            )

            if not remote_dir.exists():
                remote_dir.mkdir(parents=True, exist_ok=True)

            # ------------------------------------
            # get the metrics from the df_returns
            df_longshort_single_returns = df_longshort_returns.loc[
                df_longshort_returns["alpha"] == _alpha_name
            ].copy()
            df_longshort_single_returns.drop(columns=["alpha"], inplace=True)

            df_return = df_longshort_single_returns.copy()

            df_longshort_single_returns = df_longshort_single_returns.pivot_table(
                index="date", columns=columns_pivot, values="value", fill_value=None
            )

            df_longshort_single_returns.columns = (
                df_longshort_single_returns.columns.map(lambda x: prefix + "-".join(x))
            )

            df_longshort_metrics = qs.reports.metrics(
                returns=df_longshort_single_returns,
                rf=0.0,
                display=False,
                mode="basic",
                sep=False,
                internal="True",
                compounded=self.compound,
                periods_per_year=365,
                match_dates=False,
            )

            df_longshort_metrics = df_longshort_metrics.T
            df_longshort_metrics = df_longshort_metrics.reset_index()
            df_longshort_metrics[["group", "mode", "period"]] = df_longshort_metrics[
                "index"
            ].str.split("-", expand=True)
            df_longshort_metrics.drop(columns=["index"], inplace=True)
            df_longshort_metrics.set_index(["group", "mode", "period"], inplace=True)
            df_longshort_metrics = (
                df_longshort_metrics.stack()
                .reset_index()
                .rename(columns={"level_3": "metric", 0: "value"})
            )

            # ------------------------------------
            # add the corresponding turnover rate
            df_single_longshort_turnover = df_longshort_turnover.loc[
                df_longshort_turnover["alpha"] == _alpha_name
            ].copy()
            df_single_longshort_turnover.drop(columns=["alpha"], inplace=True)
            df_single_longshort_turnover["metric"] = "Turnover"
            df_single_longshort_turnover["value"] = df_single_longshort_turnover[
                "value"
            ].round(2)
            df_single_longshort_turnover["value"] = df_single_longshort_turnover[
                "value"
            ].astype(str)
            df_longshort_metrics = pd.concat(
                [df_longshort_metrics, df_single_longshort_turnover], ignore_index=True
            )

            # ------------------------------------
            # select the corresponding alpha
            df_alpha = df_alphas.loc[df_alphas["alpha"] == _alpha_name].copy()
            df_alpha.drop(columns=["alpha"], inplace=True)
            df_longshort_nv = df_longshort_nvs.loc[
                df_longshort_nvs["alpha"] == _alpha_name
            ].copy()
            df_longshort_nv.drop(columns=["alpha"], inplace=True)
            df_ic = df_ics.loc[df_ics["alpha"] == _alpha_name].copy()
            df_ic.drop(columns=["alpha"], inplace=True)
            df_cumic = df_cumics.loc[df_cumics["alpha"] == _alpha_name].copy()
            df_cumic.drop(columns=["alpha"], inplace=True)
            df_ic_metrics = df_ic_metrics.loc[
                df_ic_metrics["alpha"] == _alpha_name
            ].copy()
            df_ic_metrics.drop(columns=["alpha"], inplace=True)

            # ------------------------------------
            # 2) get all the stuff necessary for the stratification
            if self.groupby != GroupBy.no_group:
                columns_pivot = ["group", "layer", "period"]
                prefix = ""

            else:
                columns_pivot = ["layer", "period"]
                df_stratify_turnovers["group"] = "WHOLE"
                prefix = "WHOLE-"

            df_stratify_single_returns = df_stratify_returns.loc[
                df_stratify_returns["alpha"] == _alpha_name
            ].copy()
            df_stratify_single_returns.drop(columns=["alpha"], inplace=True)

            df_stratify_single_returns = df_stratify_single_returns.pivot_table(
                index="date", columns=columns_pivot, values="value", fill_value=None
            )

            df_stratify_single_returns.columns = df_stratify_single_returns.columns.map(
                lambda x: prefix + "-".join(x)
            )

            df_stratify_metrics = qs.reports.metrics(
                returns=df_stratify_single_returns,
                rf=0.0,
                display=False,
                mode="basic",
                sep=False,
                internal="True",
                compounded=self.compound,
                periods_per_year=365,
                match_dates=False,
            )

            df_stratify_metrics = df_stratify_metrics.T
            df_stratify_metrics = df_stratify_metrics.reset_index()
            df_stratify_metrics[["group", "layer", "period"]] = df_stratify_metrics[
                "index"
            ].str.split("-", expand=True)
            df_stratify_metrics.drop(columns=["index"], inplace=True)
            df_stratify_metrics.set_index(["group", "layer", "period"], inplace=True)
            df_stratify_metrics = (
                df_stratify_metrics.stack()
                .reset_index()
                .rename(columns={"level_3": "metric", 0: "value"})
            )

            # ------------------------------------
            # add the corresponding turnover rate
            df_single_stratify_turnover = df_stratify_turnovers.loc[
                df_stratify_turnovers["alpha"] == _alpha_name
            ].copy()
            df_single_stratify_turnover.drop(columns=["alpha"], inplace=True)
            df_single_stratify_turnover["metric"] = "Turnover"
            df_single_stratify_turnover["value"] = df_single_stratify_turnover[
                "value"
            ].round(2)
            df_single_stratify_turnover["value"] = df_single_stratify_turnover[
                "value"
            ].astype(str)
            df_stratify_metrics = pd.concat(
                [df_stratify_metrics, df_single_stratify_turnover], ignore_index=True
            )

            # ------------------------------------
            # select the corresponding alpha
            df_stratify_nv = df_stratify_nvs.loc[
                df_stratify_nvs["alpha"] == _alpha_name
            ].copy()
            df_stratify_nv.drop(columns=["alpha"], inplace=True)

            save_to_excel(
                path=remote_dir / rf"metrics.xlsx",
                df=df_longshort_metrics,
                sheet_name="LongshortMetrics",
                index=False,
            )
            save_to_excel(
                path=remote_dir / rf"metrics.xlsx",
                df=df_stratify_metrics,
                sheet_name="StratifyMetrics",
                index=False,
            )

            if generate_pdf_charts:
                self.plot_and_save_pdf(
                    data=data,
                    alpha_name=_alpha_name,
                    path=remote_dir,
                    df_alpha=df_alpha,
                    df_longshort_nv=df_longshort_nv,
                    df_longshort_metrics=df_longshort_metrics,
                    df_stratify_nv=df_stratify_nv,
                    df_stratify_metrics=df_stratify_metrics,
                    df_ic=df_ic,
                    df_cumic=df_cumic,
                    df_ic_metrics=df_ic_metrics,
                    benchmark_nv=benchmark_nv,
                )

            if not just_metrics:

                save_to_excel(
                    path=remote_dir / rf"returns.xlsx",
                    df=df_return,
                    sheet_name="LongshortReturns",
                    index=False,
                )

    def plot_and_save_pdf(
        self,
        data: StockData,
        alpha_name: str,
        path: Path,
        df_alpha: pd.DataFrame,
        df_longshort_nv: pd.DataFrame,
        df_longshort_metrics: pd.DataFrame,
        df_stratify_nv: pd.DataFrame,
        df_stratify_metrics: pd.DataFrame,
        df_ic: pd.DataFrame,
        df_cumic: pd.DataFrame,
        df_ic_metrics: pd.DataFrame,
        benchmark_nv: pd.DataFrame,
    ):
        """
        Plot the backtest results and save them as a LaTeX document with embedded plots.
        Creates comprehensive charts similar to the Strategy vs Market visualization.

        This function generates a LaTeX document with:
        1. Title page
        2. Section pages for each period
        3. Subsection pages for each group with performance charts
        4. All plots embedded as PNG images for consistent LaTeX formatting

        Files are saved as:
        - {alpha_name}_performance_report.tex: LaTeX source file
        - {alpha_name}_performance_report.pdf: Compiled PDF (if pdflatex available)
        - plots/ folder: Contains all generated plot images

        Usage:
        - Call get_nvs_metrics() with generate_pdf_charts=True
        - Or call save_longshort() with generate_pdf_charts=True
        """

        # ------------------------------------
        # 1) plot the nvs
        # 1.1) get the modes and groups
        modes = data.BacktestModes
        layers = data.layers

        # Create plots directory
        plots_dir = path / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Escape underscores in alpha name for LaTeX compatibility
        alpha_name_escaped = alpha_name.replace("_", r"\_")

        # Generate LaTeX document with proper structure
        path_latex = path / f"{alpha_name}.tex"

        before_body(path_latex=path_latex, alpha_name_escaped=alpha_name_escaped)

        # --------------------------------------------
        # Alpha distribution analysis
        part(path_latex=path_latex, text="Alpha Distribution Analysis")

        # Loop through groups as subsections
        # Get groups for this period
        if self.groupby != GroupBy.no_group:
            groupper = data.groups_names
        else:
            groupper = ["WHOLE"]
            df_cumic["group"] = "WHOLE"
            df_ic["group"] = "WHOLE"
            df_ic_metrics["group"] = "WHOLE"
            df_longshort_nv["group"] = "WHOLE"
            df_stratify_nv["group"] = "WHOLE"

        for group_idx, group in enumerate(groupper):
            # Escape underscores in group names for LaTeX compatibility
            group_escaped = group.replace("_", r"\_")

            section(path_latex=path_latex, text=group_escaped)

            # plot the distribution of df_alpha, `date`, `value`

            # Filter data for current group
            df_alpha_group = df_alpha.loc[df_alpha["group"] == group].copy()
            values = df_alpha_group["value"].dropna()

            # Create 2x2 subplot layout
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                f"Alpha Distribution Analysis - {group_escaped}",
                fontsize=16,
                fontweight="bold",
                color="#2c3e50",
                y=0.95,
            )

            # Set background colors
            fig.patch.set_facecolor("#ffffff")
            for ax in axes.flat:
                ax.set_facecolor("#f8f9fa")

            # Define winsorization levels
            winsor_levels = [
                (None, None, "Original Distribution"),
                (0.01, 0.99, "[0.01, 0.99] Winsorized"),
                (0.03, 0.97, "[0.03, 0.97] Winsorized"),
                (0.05, 0.95, "[0.05, 0.95] Winsorized"),
            ]

            # Create plots for each subplot
            for idx, (lower, upper, title) in enumerate(winsor_levels):
                row, col = idx // 2, idx % 2
                ax = axes[row, col]

                # Apply winsorization if specified (drop extreme values)
                if lower is not None and upper is not None:
                    lower_bound = values.quantile(lower)
                    upper_bound = values.quantile(upper)
                    # Drop values outside the bounds instead of clipping
                    plot_values = values[
                        (values >= lower_bound) & (values <= upper_bound)
                    ]
                    subtitle = f"{title}\n({lower_bound:.4f} to {upper_bound:.4f})"
                else:
                    plot_values = values
                    subtitle = title

                # Create histogram
                n, bins, patches = ax.hist(
                    plot_values,
                    bins=50,
                    alpha=0.7,
                    color="#3498db",
                    edgecolor="black",
                    linewidth=0.5,
                    density=True,
                    label=f"Alpha Distribution",
                )

                # Add mean line
                mean_val = plot_values.mean()
                ax.axvline(
                    mean_val,
                    color="#e74c3c",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_val:.4f}",
                )

                # Add median line
                median_val = plot_values.median()
                ax.axvline(
                    median_val,
                    color="#2ecc71",
                    linestyle="--",
                    linewidth=2,
                    label=f"Median: {median_val:.4f}",
                )

                # Add zero line for reference
                ax.axvline(
                    0,
                    color="#34495e",
                    linestyle="-",
                    linewidth=1,
                    alpha=0.5,
                    label="Zero Reference",
                )

                # Add statistics text box
                stats_text = f"""Stats:
Mean: {mean_val:.4f}
Median: {median_val:.4f}
Std: {plot_values.std():.4f}
Min: {plot_values.min():.4f}
Max: {plot_values.max():.4f}
Count: {len(plot_values):,}"""

                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    fontsize=8,
                    fontfamily="monospace",
                )

                # Set labels and title
                ax.set_xlabel(
                    "Alpha Value", fontsize=10, fontweight="bold", color="#2c3e50"
                )
                ax.set_ylabel(
                    "Density", fontsize=10, fontweight="bold", color="#2c3e50"
                )
                ax.set_title(subtitle, fontsize=11, fontweight="bold", color="#2c3e50")

                # Add grid and legend
                ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

                # Format x-axis
                ax.tick_params(axis="x", rotation=0)

            # Adjust layout and save
            plt.tight_layout()

            # Generate plot filename for this group
            plot_filename = f"alpha_distribution_{group_idx}.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()

            # Create individual subplot files for the grid layout
            subplot_paths = []
            for idx, (lower, upper, title) in enumerate(winsor_levels):
                # Create individual subplot
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.set_facecolor("#f8f9fa")

                # Apply winsorization if specified (drop extreme values)
                if lower is not None and upper is not None:
                    lower_bound = values.quantile(lower)
                    upper_bound = values.quantile(upper)
                    # Drop values outside the bounds instead of clipping
                    plot_values = values[
                        (values >= lower_bound) & (values <= upper_bound)
                    ]
                else:
                    plot_values = values

                # Create histogram
                ax.hist(
                    plot_values,
                    bins=50,
                    alpha=0.7,
                    color="#3498db",
                    edgecolor="black",
                    linewidth=0.5,
                    density=True,
                )

                # Add mean and median lines
                mean_val = plot_values.mean()
                median_val = plot_values.median()
                ax.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=2)
                ax.axvline(median_val, color="#2ecc71", linestyle="--", linewidth=2)
                ax.axvline(0, color="#34495e", linestyle="-", linewidth=1, alpha=0.5)

                # Set title and labels
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.set_xlabel("Alpha Value", fontsize=9)
                ax.set_ylabel("Density", fontsize=9)
                ax.grid(True, alpha=0.3)

                # Save individual subplot
                subplot_filename = f"alpha_dist_{group_idx}_subplot_{idx}.png"
                subplot_path = plots_dir / subplot_filename
                plt.savefig(
                    subplot_path, dpi=300, bbox_inches="tight", facecolor="white"
                )
                plt.close()

                subplot_paths.append(subplot_path)

            # Insert the grid of plots into LaTeX
            include_multiple_plots_grid(
                path_latex=path_latex,
                plot_paths=subplot_paths,
                caption=f"Alpha Distribution Analysis for {group_escaped} - Original vs Winsorized Distributions",
            )

        # Loop through periods as main sections

        # --------------------------------------------
        # IC Analysis
        newpage(path_latex=path_latex)
        part(path_latex=path_latex, text="IC Analysis")

        # Loop through periods as main sections
        for period_idx, period in enumerate(data.periods):
            # Escape underscores in period names for LaTeX compatibility
            period_escaped = period.replace("_", r"\_")
            # Section title page
            section(path_latex=path_latex, text=rf"{period_escaped} Forecast")

            # Get groups for this period
            if self.groupby != GroupBy.no_group:
                groupper = data.groups_names
            else:
                groupper = ["WHOLE"]
                df_ic["group"] = "WHOLE"
                df_cumic["group"] = "WHOLE"

            # Loop through groups as subsections
            for group_idx, group in enumerate(groupper):
                # Escape underscores in group names for LaTeX compatibility
                group_escaped = group.replace("_", r"\_")

                subsection(path_latex=path_latex, text=group_escaped)

                # Loop through layers as subsubsections
                for layer_idx, layer in enumerate(data.ic_layers):
                    # Escape underscores in layer names for LaTeX compatibility
                    layer_escaped = layer.replace("_", r"\_")

                    subsubsection(path_latex=path_latex, text=layer_escaped)

                    # plot the ic graph, cumic and ic
                    # Generate plot filename for this group and layer
                    plot_filename = f"ic_plot_{period_idx}_{group_idx}_{layer_idx}.png"
                    plot_path = plots_dir / plot_filename

                    # Create the combined plot for IC and cumulative IC with dual y-axes
                    fig, ax1 = plt.subplots(figsize=(12, 8))

                    # Set background colors
                    fig.patch.set_facecolor("#ffffff")
                    ax1.set_facecolor("#f8f9fa")

                    # Create second y-axis for cumulative IC
                    ax2 = ax1.twinx()

                    # Filter data for current group, period, and layer
                    df_ic_group = df_ic.loc[
                        (df_ic["group"] == group)
                        & (df_ic["period"] == period)
                        & (df_ic["layer"] == layer)
                    ].copy()
                    df_cumic_group = df_cumic.loc[
                        (df_cumic["group"] == group)
                        & (df_cumic["period"] == period)
                        & (df_cumic["layer"] == layer)
                    ].copy()

                    min_date = min(
                        df_ic_group["date"].min(), df_cumic_group["date"].min()
                    )
                    max_date = max(
                        df_ic_group["date"].max(), df_cumic_group["date"].max()
                    )

                    # Set consistent x-axis limits with more padding for bars
                    ax1.set_xlim(
                        min_date - pd.Timedelta(days=20),
                        max_date + pd.Timedelta(days=20),
                    )

                    # Top subplot: Monthly average IC bar plot
                    df_ic_group["date"] = pd.to_datetime(df_ic_group["date"])
                    df_ic_group["month"] = df_ic_group["date"].dt.to_period("M")

                    # Calculate monthly average IC
                    monthly_ic = df_ic_group.groupby("month")["value"].mean()

                    # Create bar plot using actual dates for proper alignment
                    month_dates = [
                        pd.to_datetime(str(m) + "-01") for m in monthly_ic.index
                    ]
                    ic_values = monthly_ic.values

                    # Color bars based on positive/negative values
                    colors = ["#2ecc71" if x >= 0 else "#e74c3c" for x in ic_values]

                    # Calculate bar width based on date range - make it smaller to avoid overlap
                    bar_width = pd.Timedelta(
                        days=10
                    )  # Smaller width to prevent x-axis overlap

                    bars = ax1.bar(
                        month_dates,
                        ic_values,
                        width=bar_width,
                        color=colors,
                        alpha=0.7,
                        edgecolor="black",
                        linewidth=0.5,
                        label="Monthly IC",
                    )

                    # Bar value labels removed since LaTeX caption provides this information
                    ax1.set_ylabel(
                        "Monthly IC Value",
                        fontsize=10,
                        fontweight="bold",
                        color="#2c3e50",
                    )
                    ax1.set_xlabel(
                        "Date", fontsize=10, fontweight="bold", color="#2c3e50"
                    )
                    ax1.grid(True, alpha=0.3)
                    ax1.axhline(
                        y=0, color="black", linestyle="-", alpha=0.5, linewidth=0.5
                    )

                    # Bottom subplot: Daily cumulative IC time series
                    df_cumic_group["date"] = pd.to_datetime(df_cumic_group["date"])
                    df_cumic_group = df_cumic_group.sort_values("date")

                    # Plot cumulative IC on right y-axis
                    ax2.plot(
                        df_cumic_group["date"],
                        df_cumic_group["value"],
                        linewidth=2,
                        color="#3498db",
                        marker="o",
                        markersize=3,
                        markevery=len(df_cumic_group) // 20,
                        label="Cumulative IC",
                    )

                    ax2.set_ylabel(
                        "Cumulative IC", fontsize=10, fontweight="bold", color="#3498db"
                    )
                    ax2.grid(False)  # Don't show grid for right y-axis
                    ax2.axhline(
                        y=0, color="#3498db", linestyle="-", alpha=0.5, linewidth=0.5
                    )

                    # Format x-axis dates with better spacing to reduce overlap
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                    ax1.xaxis.set_major_locator(
                        mdates.MonthLocator(interval=1)
                    )  # Show every 2nd month to reduce overlap
                    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

                    # Improve x-axis label spacing
                    ax1.tick_params(axis="x", which="major", pad=15)

                    # Add clean legends without statistics
                    ax1.legend(loc="upper left", framealpha=0.9)
                    ax2.legend(loc="upper right", framealpha=0.9)

                    # Ensure both y-axes are properly scaled and visible
                    ax1.tick_params(axis="y", colors="#2c3e50")
                    ax2.tick_params(axis="y", colors="#3498db")

                    # Align both y-axes at 0
                    y1_min, y1_max = ax1.get_ylim()
                    y2_min, y2_max = ax2.get_ylim()

                    # Find the range that includes 0 for both axes
                    y1_range = max(abs(y1_min), abs(y1_max))
                    y2_range = max(abs(y2_min), abs(y2_max))

                    # Set both axes to have 0 centered and similar scale
                    ax1.set_ylim(-y1_range, y1_range)
                    ax2.set_ylim(-y2_range, y2_range)

                    # Adjust layout and save
                    plt.tight_layout()
                    plt.savefig(
                        plot_path, dpi=300, bbox_inches="tight", facecolor="white"
                    )
                    plt.close()

                    # Insert the plot into LaTeX
                    includegraph(
                        path_latex=path_latex,
                        path_graph=plot_path,
                        caption=f"IC Analysis for {group_escaped} ({period_escaped}) - {layer_escaped}",
                    )

                    # Loop through modes to add metrics tables as subsubsections
                    df_single_ic_metrics = df_ic_metrics.loc[
                        (df_ic_metrics["group"] == group)
                        & (df_ic_metrics["period"] == period)
                        & (df_ic_metrics["layer"] == layer)
                    ].copy()

                    df_single_ic_metrics = df_single_ic_metrics.loc[
                        df_single_ic_metrics["metrics"].isin(
                            [
                                "MEAN",
                                "POSITIVE_RATIO",
                                "STD",
                                "RISK_ADJUSTED_IC",
                                "T_STATS",
                                "P_VALUE",
                                "SKEW",
                                "KURTOSIS",
                            ]
                        )
                    ]
                    df_single_ic_metrics.drop(
                        columns=["group", "layer", "period", "parent"], inplace=True
                    )
                    df_single_ic_metrics["value"] = (
                        df_single_ic_metrics["value"].round(4).astype(str)
                    )

                    include_ic_metrics_table(
                        path_latex=path_latex,
                        df_metrics=df_single_ic_metrics,
                        caption="IC Metrics",
                    )

                    # End of layer loop
                # End of group loop
                newpage(path_latex=path_latex)
            newpage(path_latex=path_latex)
        newpage(path_latex=path_latex)

        # --------------------------------------------
        # Long Short Backtest
        part(path_latex=path_latex, text="Long Short Backtest")

        # Loop through periods as main sections
        for period_idx, period in enumerate(data.periods):
            # Escape underscores in period names for LaTeX compatibility
            period_escaped = period.replace("_", r"\_")
            # Section title page
            section(path_latex=path_latex, text=rf"{period_escaped} Rebalance")

            # Get groups for this period
            if self.groupby != GroupBy.no_group:
                groupper = data.groups_names
            else:
                groupper = ["WHOLE"]
                df_longshort_nv["group"] = "WHOLE"

            # Loop through groups as subsections
            for group_idx, group in enumerate(groupper):
                # Escape underscores in group names for LaTeX compatibility
                group_escaped = group.replace("_", r"\_")

                subsection(path_latex=path_latex, text=group_escaped)

                # Generate plot filename for this group
                plot_filename = f"plot_{period_idx}_{group_idx}.png"
                plot_path = plots_dir / plot_filename

                # Create the combined plot for net values only
                fig, ax = plt.subplots(figsize=(12, 8))

                # Set background color
                fig.patch.set_facecolor("#ffffff")
                ax.set_facecolor("#f8f9fa")

                # Define colors for different modes
                mode_colors = [
                    "#e74c3c",
                    "#3498db",
                    "#2ecc71",
                    "#f39c12",
                    "#9b59b6",
                    "#e67e22",
                ]

                # Plot net value for each mode on the main axis
                for mode_idx, mode in enumerate(modes):
                    df_single_nv = df_longshort_nv.loc[
                        (df_longshort_nv["mode"] == mode)
                        & (df_longshort_nv["group"] == group)
                        & (df_longshort_nv["period"] == period)
                    ].copy()

                    df_single_nv.drop(columns=["mode", "group", "period"], inplace=True)

                    # Use modulo to cycle through colors if more modes than colors
                    color = mode_colors[mode_idx % len(mode_colors)]

                    ax.plot(
                        df_single_nv["date"],
                        np.log(df_single_nv["value"]),
                        linewidth=2,
                        color=color,
                        label=f"{mode}",
                        marker="o",
                        markersize=2,
                        markevery=len(df_single_nv) // 20,
                    )

                # Add benchmark if available
                ax.plot(
                    benchmark_nv["time"],
                    np.log(benchmark_nv["ret"]),
                    "--",
                    color="#34495e",
                    linewidth=2,
                    label="Market (Equal-weight)",
                    alpha=0.8,
                )

                # Set labels and styling
                ax.set_ylabel(
                    "Net Value (log scale)",
                    fontsize=11,
                    fontweight="bold",
                    color="#2c3e50",
                )
                ax.set_xlabel("Date", fontsize=11, fontweight="bold", color="#2c3e50")

                # No title since we have LaTeX titles

                # Grid and styling for the main axis
                ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(colors="#7f8c8d", labelsize=9)

                # Format x-axis dates
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

                # Add legend
                ax.legend(
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    fontsize=10,
                    loc="upper left",
                )

                # Save plot and close
                plt.tight_layout(pad=1.5)
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                # Add figure to LaTeX
                includegraph(
                    path_latex=path_latex, path_graph=plot_path, caption="Longshort NV"
                )

                # Loop through modes to add metrics tables as subsubsections
                df_single_metrics = df_longshort_metrics.loc[
                    (df_longshort_metrics["group"] == group)
                    & (df_longshort_metrics["period"] == period)
                ].copy()

                df_single_metrics = df_single_metrics.loc[
                    df_single_metrics["metric"].isin(
                        [
                            "Start Period",
                            "End Period",
                            "CAGR﹪",
                            "Sharpe",
                            "Max Drawdown",
                            "Turnover",
                        ]
                    )
                ]
                df_single_metrics.drop(columns=["group", "period"], inplace=True)

                df_single_metrics = df_single_metrics.pivot_table(
                    index="mode", columns="metric", values="value", aggfunc="first"
                ).reset_index()

                df_single_metrics = df_single_metrics[
                    [
                        "mode",
                        "Start Period",
                        "End Period",
                        "CAGR﹪",
                        "Sharpe",
                        "Max Drawdown",
                        "Turnover",
                    ]
                ]

                include_longshort_metrics_table(
                    path_latex=path_latex,
                    df_metrics=df_single_metrics,
                    caption="Metrics",
                )

                newpage(path_latex=path_latex)
            newpage(path_latex=path_latex)
        newpage(path_latex=path_latex)

        # --------------------------------------------
        # Stratify Backtest
        newpage(path_latex=path_latex)
        part(path_latex=path_latex, text="Stratify Backtest")

        # Loop through periods as main sections
        for period_idx, period in enumerate(data.periods):
            # Escape underscores in period names for LaTeX compatibility
            period_escaped = period.replace("_", r"\_")
            # Section title page
            section(path_latex=path_latex, text=rf"{period_escaped} Rebalance")

            # Get groups for this period
            if self.groupby != GroupBy.no_group:
                groupper = data.groups_names
            else:
                groupper = ["WHOLE"]
                df_stratify_nv["group"] = "WHOLE"

            # Loop through groups as subsections
            for group_idx, group in enumerate(groupper):
                # Escape underscores in group names for LaTeX compatibility
                group_escaped = group.replace("_", r"\_")

                subsection(path_latex=path_latex, text=group_escaped)

                # Generate plot filename for this group
                plot_filename = f"stratify_plot_{period_idx}_{group_idx}.png"
                plot_path = plots_dir / plot_filename

                # Create the combined plot for all layers
                fig, ax = plt.subplots(figsize=(12, 8))

                # Set background color
                fig.patch.set_facecolor("#ffffff")
                ax.set_facecolor("#f8f9fa")

                # Define colors for different layers
                layer_colors = [
                    "#e74c3c",
                    "#3498db",
                    "#2ecc71",
                    "#f39c12",
                    "#9b59b6",
                    "#e67e22",
                ]

                # Plot net value for each layer on the same axis
                for layer_idx, layer in enumerate(layers):
                    df_single_nv = df_stratify_nv.loc[
                        (df_stratify_nv["layer"] == layer)
                        & (df_stratify_nv["group"] == group)
                        & (df_stratify_nv["period"] == period)
                    ].copy()

                    if not df_single_nv.empty:
                        df_single_nv.drop(
                            columns=["layer", "group", "period"], inplace=True
                        )

                        # Use modulo to cycle through colors if more layers than colors
                        color = layer_colors[layer_idx % len(layer_colors)]

                        ax.plot(
                            df_single_nv["date"],
                            np.log(df_single_nv["value"]),
                            linewidth=2,
                            color=color,
                            label=f"{layer}",
                            marker="o",
                            markersize=2,
                            markevery=len(df_single_nv) // 20,
                        )

                # Add benchmark if available
                ax.plot(
                    benchmark_nv["time"],
                    np.log(benchmark_nv["ret"]),
                    "--",
                    color="#34495e",
                    linewidth=2,
                    label="Market (Equal-weight)",
                    alpha=0.8,
                )

                # Set labels and styling
                ax.set_ylabel(
                    "Net Value (log scale)",
                    fontsize=11,
                    fontweight="bold",
                    color="#2c3e50",
                )
                ax.set_xlabel("Date", fontsize=11, fontweight="bold", color="#2c3e50")

                # Grid and styling for the main axis
                ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(colors="#7f8c8d", labelsize=9)

                # Format x-axis dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

                # Add legend
                ax.legend(loc="upper left", framealpha=0.9)

                # Adjust layout and save
                plt.tight_layout()
                plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
                plt.close()

                # Insert the plot into LaTeX
                includegraph(
                    path_latex=path_latex,
                    path_graph=plot_path,
                    caption=f"Stratify Net Value for {group_escaped} ({period_escaped})",
                )

                # Add metrics table for all layers in this group
                df_group_metrics = df_stratify_metrics.loc[
                    (df_stratify_metrics["group"] == group)
                    & (df_stratify_metrics["period"] == period)
                ].copy()

                # Process metrics for the table
                df_group_metrics = df_group_metrics.pivot_table(
                    index="layer",
                    columns="metric",
                    values="value",
                    aggfunc="first",
                ).reset_index()

                df_group_metrics = df_group_metrics[
                    [
                        "layer",
                        "Start Period",
                        "End Period",
                        "CAGR﹪",
                        "Sharpe",
                        "Max Drawdown",
                        "Turnover",
                    ]
                ]

                include_stratify_metrics_table(
                    path_latex=path_latex,
                    df_metrics=df_group_metrics,
                    caption=f"Metrics for {group_escaped} ({period_escaped})",
                )

                newpage(path_latex=path_latex)
            newpage(path_latex=path_latex)
        newpage(path_latex=path_latex)

        # LaTeX document footer
        end(path_latex=path_latex)

        logging.info(f"-> saved LaTeX performance report")

        # Function to clean up intermediate files
        def cleanup_files():
            # Remove LaTeX source file
            if path_latex.exists():
                path_latex.unlink()

            # Remove plots directory and all its contents
            if plots_dir.exists():
                import shutil

                shutil.rmtree(plots_dir)

            # Remove other LaTeX auxiliary files
            aux_files = [
                path_latex.with_suffix(".aux"),
                path_latex.with_suffix(".log"),
                path_latex.with_suffix(".out"),
                path_latex.with_suffix(".toc"),
            ]
            for aux_file in aux_files:
                if aux_file.exists():
                    aux_file.unlink()

            logging.info("-> cleanup complete")

        # Try to compile the LaTeX document using pdflatex for faster compilation
        # pdflatex is faster and handles table of contents well

        # First compilation - generates initial .toc file (usually empty)
        result1 = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(path_latex)],
            cwd=str(path),
            capture_output=True,
            text=True,
        )

        if result1.returncode == 0:
            logging.info(f"-> first pdflatex compilation successful")

            # Second compilation - processes sections and updates .toc file
            result2 = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", str(path_latex)],
                cwd=str(path),
                capture_output=True,
                text=True,
            )

            if result2.returncode == 0:
                logging.info(f"-> second pdflatex compilation successful")
                cleanup_files()
            else:
                logging.error(
                    f"Second pdflatex compilation failed with return code: {result2.returncode}"
                )
                # Clean up anyway since compilation failed
                cleanup_files()
        else:
            logging.error(
                f"Manual compilation command: cd {path} && pdflatex {path_latex.name} && pdflatex {path_latex.name} && pdflatex {path_latex.name}"
            )
            # Clean up anyway since compilation failed
            cleanup_files()

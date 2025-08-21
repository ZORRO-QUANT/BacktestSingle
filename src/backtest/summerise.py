from .constants import *
from .utils import *

path_general = Path(load_config("path.yaml")["general"])


class Summerizer:
    def __init__(
        self, data_source: DataSource, groupby: GroupBy, parent_alpha: Alpha
    ) -> None:

        self.data_source = data_source
        self.groupby = groupby
        self.parent_alpha = parent_alpha

        self.alphas = self._get_children()

    def _get_children(self):

        # ------------------------------------
        # get all the children alphas which belong to the parent
        path_folder = (
            path_general
            / self.data_source.exchange.name
            / self.data_source.universe.name
            / "Alphas"
            / self.parent_alpha.freq
            / self.parent_alpha.category.name
        )

        alpha_names = [
            full_name.stem.split(">")[0]
            for full_name in list(path_folder.glob(f"*>{self.parent_alpha.name}.csv"))
        ]

        # ------------------------------------
        # create all the alphas from the alpha_names
        alphas = [
            Alpha(
                category=self.parent_alpha.category,
                name=alpha,
                freq=self.parent_alpha.freq,
                parent=self.parent_alpha.name,
            )
            for alpha in alpha_names
        ]

        if self.parent_alpha.parent != "":
            alphas.append(self.parent_alpha)

        return alphas

    def summerise(self):

        self._summerise_longshort()
        self._summerise_stratify()

    def _summerise_longshort(self):

        # -------------------------------
        # 1) get folder path and loop

        dfs = []

        for alpha in self.alphas:

            alpha_path = (
                path_general
                / self.data_source.exchange.name
                / self.data_source.universe.name
                / "Backtest"
                / self.groupby.name
                / self.parent_alpha.category.name
                / f"{alpha.name}>{alpha.parent}"
                / "metrics.xlsx"
            )

            df_longshort = pd.read_excel(
                alpha_path, sheet_name="LongshortMetrics", engine="openpyxl"
            )
            df_longshort = df_longshort.loc[df_longshort["metric"] == "Sharpe"]
            df_longshort["alpha"] = alpha.name

            dfs.append(df_longshort)

        df_total = pd.concat(dfs, ignore_index=True)

        parent_summary_path = (
            path_general
            / self.data_source.exchange.name
            / self.data_source.universe.name
            / "Backtest"
            / self.groupby.name
            / self.parent_alpha.category.name
            / f"{self.parent_alpha.name}.xlsx"
        )

        save_to_excel(
            path=parent_summary_path,
            df=df_total,
            sheet_name="LongshortSummary",
            index=False,
        )

        logging.info(rf"successfully saved {self.parent_alpha.name} longshort summary")

    def _summerise_stratify(self):

        # -------------------------------
        # 1) get folder path and loop

        dfs = []

        for alpha in self.alphas:

            alpha_path = (
                path_general
                / self.data_source.exchange.name
                / self.data_source.universe.name
                / "Backtest"
                / self.groupby.name
                / self.parent_alpha.category.name
                / f"{alpha.name}>{alpha.parent}"
                / "metrics.xlsx"
            )

            df_stratify = pd.read_excel(
                alpha_path, sheet_name="StratifyMetrics", engine="openpyxl"
            )
            df_stratify = df_stratify.loc[df_stratify["metric"] == "Sharpe"]
            df_stratify["alpha"] = alpha.name

            dfs.append(df_stratify)

        df_total = pd.concat(dfs, ignore_index=True)

        parent_summary_path = (
            path_general
            / self.data_source.exchange.name
            / self.data_source.universe.name
            / "Backtest"
            / self.groupby.name
            / self.parent_alpha.category.name
            / f"{self.parent_alpha.name}.xlsx"
        )

        save_to_excel(
            path=parent_summary_path,
            df=df_total,
            sheet_name="StratifySummary",
            index=False,
        )

        logging.info(rf"successfully saved {self.parent_alpha.name} stratify summary")

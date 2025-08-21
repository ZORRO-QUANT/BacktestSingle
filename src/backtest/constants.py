from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from typing import Literal, Optional


class Category(IntEnum):
    size_1d = auto()
    liquidity_1d = auto()
    momentum_1d = auto()
    pv_1d = auto()
    volatility_1d = auto()
    imbalance_1d = auto()
    technical_1d = auto()

    imbalance_hf_1d = auto()
    liquidity_hf_1d = auto()
    momentum_hf_1d = auto()
    pv_hf_1d = auto()
    volatility_hf_1d = auto()

    liquidity_1h = auto()
    momentum_1h = auto()
    pv_1h = auto()
    volatility_1h = auto()
    game_1h = auto()
    imbalance_1h = auto()

    imbalance_hf_1h = auto()
    liquidity_hf_1h = auto()
    momentum_hf_1h = auto()
    pv_hf_1h = auto()
    volatility_hf_1h = auto()

    development_1d = auto()
    development_1h = auto()

    disagree_contract = auto()
    game_contract = auto()
    liquidity_contract = auto()
    momentum_contract = auto()
    turnover_contract = auto()
    volatility_contract = auto()

    group_MIDDLE_pool_10_seed_44_policy_LSTM_target_5 = auto()


@dataclass
class DataSources:
    factor: "DataSource"
    kline: "DataSource"
    group: "DataSource"


@dataclass
class DataSource:
    exchange: "Exchange"
    universe: "Universe"
    freq: Literal["1h", "1d"]


class Exchange(IntEnum):
    Okx = auto()
    Binance = auto()
    Crossover = auto()


class Universe(IntEnum):
    spot = auto()
    perp = auto()


class Benchmark(IntEnum):
    whole = auto()
    btc = auto()


class Evaluation(IntEnum):
    ics = auto()
    cumics = auto()
    alphas = auto()
    ic_metrics = auto()
    ret_metrics = auto()
    nvs = auto()
    returns = auto()
    drawdowns = auto()
    turnover = auto()


@dataclass
class Alpha:
    category: "Category"
    name: str
    aggregations: dict = field(default_factory=dict)
    freq: Optional[Literal["1h", "1d"]] = "1d"
    parent: Optional[str] = None

    @property
    def path(self) -> Path:
        if self.parent is None:
            return Path(f"Alphas/{self.freq}/{self.category.name}/{self.name}.csv")
        else:
            return Path(
                f"Alphas/{self.freq}/{self.category.name}/{self.name}>{self.parent}.csv"
            )


@dataclass
class Parent:
    category: "Category"
    name: str
    freq: Optional[Literal["1h", "1d"]] = "1d"


class GroupBy(IntEnum):
    amount_quarter_perp_3 = auto()
    amount_quarter_spot_3 = auto()
    mcap_quarter_spot_3 = auto()
    amount_quarter_spot_4 = auto()
    no_group = auto()


class Amount2Group(IntEnum):
    BIG = 0
    SMALL = 1


class Amount3Group(IntEnum):
    BIG = 0
    MIDDLE = 1
    SMALL = 2


class Liquidity3Group(IntEnum):
    HIGH = 0
    MIDDLE = 1
    LOW = 2


class Amount4Group(IntEnum):
    SUPER_BIG = 0
    BIG = 1
    SMALL = 2
    SUPER_SMALL = 3


class Metrics_IC(IntEnum):
    MEAN = 0
    POSITIVE_RATIO = 1
    STD = 2
    RISK_ADJUSTED_IC = 3
    T_STATS = 4
    P_VALUE = 5
    SKEW = 6
    KURTOSIS = 7


class Metrics_Stratify(IntEnum):
    MEAN = 0
    WIN_RATE = 1
    STD = 2
    RISK_ADJUSTED_MEAN = 3
    T_STATS = 4
    P_VALUE = 5
    TURNOVER = 6


class BacktestModes(IntEnum):
    LONG = 0
    SHORT = 1
    LONG_SHORT = 2

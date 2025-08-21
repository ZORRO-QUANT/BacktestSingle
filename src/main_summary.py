from os import name

from backtest import Summerizer
from backtest.constants import *

if __name__ == "__main__":

    data_source = DataSource(
        exchange=Exchange.Binance, universe=Universe.spot, freq="1h"
    )
    groupby = GroupBy.amount_quarter_spot_3

    parent_alpha = Alpha(
        category=Category.momentum_1d,
        name="mmt_groupbeta",
        freq="1d",
        parent="",
    )

    instance = Summerizer(
        data_source=data_source, parent_alpha=parent_alpha, groupby=groupby
    )
    instance.summerise()

import configparser
from typing import Optional, Tuple

import torch

from .constants import BacktestModes
from .utils import *

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the configuration file
commission = load_config("backtest.yaml")["commission"]
long_upper = load_config("backtest.yaml")["long"]["upper"]
long_lower = load_config("backtest.yaml")["long"]["lower"]
short_upper = load_config("backtest.yaml")["short"]["upper"]
short_lower = load_config("backtest.yaml")["short"]["lower"]


def rank_information_coefficient(
    alphas: torch.Tensor,  # (dates, alphas, stocks)
    returns: torch.Tensor,  # (dates, periods, stocks)
    groups: Optional[torch.Tensor] = None,  # (dates, stocks)
    by_group: bool = False,
) -> torch.Tensor:
    """
    Computes daily rankIC for each alpha and period using matrix operations (no loops).

    Parameters
    ----------
    alphas : torch.Tensor
        Shape (dates, stocks, alphas). The alpha values for each date and stock.
    returns : torch.Tensor
        Shape (dates, stocks, periods). The forward returns for each date and stock.

    Returns
    -------
    ic_values : torch.Tensor
        Shape (dates, periods, alphas, 1). The daily IC values between alphas and forward returns if not grouping
        Shape (dates, periods, groups, alphas, 1). If grouping
    """
    if by_group and (groups is None):
        raise ValueError("please provide the groups tensor")

    # Get the shape
    _, num_alphas, _ = alphas.shape
    _, num_periods, _ = returns.shape

    # Expand the dimensions of alphas and returns for broadcasting
    alphas_exp = alphas.unsqueeze(2).expand(
        -1, -1, num_periods, -1
    )  # (dates, alphas, periods_fake, stocks)
    returns_exp = returns.unsqueeze(1).expand(
        -1, num_alphas, -1, -1
    )  # (dates, alphas_fake, periods, stocks)

    # Do the mask
    non_nan_mask = ~torch.isnan(alphas_exp) & ~torch.isnan(returns_exp)
    alphas_exp = torch.where(non_nan_mask, alphas_exp, torch.nan)
    returns_exp = torch.where(non_nan_mask, returns_exp, torch.nan)

    def compute_ics(
        alphas_exp_tensor: torch.Tensor, returns_exp_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Spearman IC by calculating the Spearman correlation.

        :param alphas_exp_tensor: tensor(dates, stocks, periods_fake, alphas)
        :param returns_exp_tensor: tensor(dates, stocks, periods, alphas_fake)
        :return: tensor(dates, alphas, periods)
        """

        # Step 1: Rank the data along the 'stocks' dimension (dim=1)
        # replace the nans with inf, so it will be out of the sorting mechanism

        alphas_rank = (
            torch.where(torch.isnan(alphas_exp_tensor), torch.inf, alphas_exp_tensor)
            .argsort(dim=-1)
            .argsort(dim=-1)
            .float()
        )
        returns_rank = (
            torch.where(torch.isnan(returns_exp_tensor), torch.inf, returns_exp_tensor)
            .argsort(dim=-1)
            .argsort(dim=-1)
            .float()
        )

        alphas_rank = torch.where(
            torch.isnan(alphas_exp_tensor), torch.nan, alphas_rank
        )
        returns_rank = torch.where(
            torch.isnan(returns_exp_tensor), torch.nan, returns_rank
        )

        # Step 2: Compute means of ranked alphas and ranked returns over the 'stocks' dimension
        alphas_mean = torch.nanmean(
            alphas_rank, dim=-1, keepdim=True
        )  # (dates, alphas, periods_fake, 1)
        returns_mean = torch.nanmean(
            returns_rank, dim=-1, keepdim=True
        )  # (dates, alphas_fake, periods, 1)

        # Step 3: Compute covariance of ranks
        # (dates, alphas, periods_fake)
        cov = torch.nanmean(
            (alphas_rank - alphas_mean) * (returns_rank - returns_mean), dim=-1
        )

        # Step 4: Compute standard deviations of ranks
        alphas_std = nanstd(alphas_rank, dim=-1)  # (dates, alphas, periods_fake)
        returns_std = nanstd(returns_rank, dim=-1)  # (dates, alphas_fake, periods)

        # Step 5: Spearman correlation = Pearson correlation of ranked data
        ic_values = cov / (alphas_std * returns_std)

        del (
            alphas_rank,
            returns_rank,
            alphas_mean,
            returns_mean,
            cov,
            alphas_std,
            returns_std,
        )
        torch.cuda.empty_cache()

        return ic_values  # (dates, alphas, periods)

    if not by_group:

        # if not by group, directly compute the ics without masking
        ic_values = compute_ics(alphas_exp, returns_exp)

        # unsqueeze the last dim
        ic_values = ic_values.unsqueeze(dim=-1)

        return ic_values  # (dates, alphas, periods, 1)

    else:

        # if we need to group, first we should create masks by the unique groups
        non_nan_group = groups[~torch.isnan(groups)]
        unique_groups = torch.unique(non_nan_group)

        # expand the group
        # (dates, alphas_fake, periods_fake, stocks)
        group_exp = (
            groups.unsqueeze(1).unsqueeze(1).expand(-1, num_alphas, num_periods, -1)
        )

        groups_lst = []
        # loop over the unqiue groups and create mask one by one
        for group in unique_groups:
            # create a mask where group_exp matches the current unique group
            mask = group_exp == group

            alphas_exp_mask = torch.where(mask, alphas_exp, torch.nan)
            returns_exp_mask = torch.where(mask, returns_exp, torch.nan)

            ic_values_group = compute_ics(alphas_exp_mask, returns_exp_mask)
            groups_lst.append(ic_values_group)

        ic_values_groups = torch.stack(groups_lst, dim=1)

        # unsqueeze the last dim
        ic_values_groups = ic_values_groups.unsqueeze(dim=-1)

        return ic_values_groups  # (dates, groups, alphas, periods, 1)


def information_coefficient(
    alphas: torch.Tensor,  # (dates, alphas, stocks)
    returns: torch.Tensor,  # (dates, periods, stocks)
    groups: Optional[torch.Tensor] = None,  # (dates, stocks)
    by_group: bool = False,
) -> torch.Tensor:
    """
    Computes daily IC for each alpha and period using matrix operations (no loops).

    Parameters
    ----------
    alphas : torch.Tensor
        Shape (dates, stocks, alphas). The alpha values for each date and stock.
    returns : torch.Tensor
        Shape (dates, stocks, periods). The forward returns for each date and stock.

    Returns
    -------
    ic_values : torch.Tensor
        Shape (dates, periods, alphas, 1). The daily IC values between alphas and forward returns if not grouping
        Shape (dates, periods, groups, alphas, 1). If grouping
    """
    if by_group and (groups is None):
        raise ValueError("please provide the groups tensor")

    # Get the shape
    num_dates, num_alphas, num_stocks = alphas.shape
    _, num_periods, _ = returns.shape

    # Expand the dimensions of alphas and returns for broadcasting
    alphas_exp = alphas.unsqueeze(2).expand(
        -1, -1, num_periods, -1
    )  # (dates, alphas, periods_fake, stocks)
    returns_exp = returns.unsqueeze(1).expand(
        -1, num_alphas, -1, -1
    )  # (dates, alphas_fake, periods, stocks)

    # Do the mask
    non_nan_mask = ~torch.isnan(alphas_exp) & ~torch.isnan(returns_exp)
    alphas_exp = torch.where(non_nan_mask, alphas_exp, torch.nan)
    returns_exp = torch.where(non_nan_mask, returns_exp, torch.nan)

    def compute_ics(
        alphas_exp_tensor: torch.Tensor, returns_exp_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Spearman IC by calculating the Spearman correlation.

        :param alphas_exp_tensor: tensor(dates, stocks, periods_fake, alphas)
        :param returns_exp_tensor: tensor(dates, stocks, periods, alphas_fake)
        :return: tensor(dates, alphas, periods)
        """

        # Step 1: Rank the data along the 'stocks' dimension (dim=1)
        # replace the nans with inf, so it will be out of the sorting mechanism

        # Step 2: Compute means of ranked alphas and ranked returns over the 'stocks' dimension
        alphas_mean = torch.nanmean(
            alphas_exp_tensor, dim=-1, keepdim=True
        )  # (dates, alphas, periods_fake, 1)
        returns_mean = torch.nanmean(
            returns_exp_tensor, dim=-1, keepdim=True
        )  # (dates, alphas_fake, periods, 1)

        # Step 3: Compute covariance of ranks
        # (dates, alphas, periods_fake)
        cov = torch.nanmean(
            (alphas_exp_tensor - alphas_mean) * (returns_exp_tensor - returns_mean),
            dim=-1,
        )

        # Step 4: Compute standard deviations of ranks
        alphas_std = nanstd(alphas_exp_tensor, dim=-1)  # (dates, alphas, periods_fake)
        returns_std = nanstd(
            returns_exp_tensor, dim=-1
        )  # (dates, alphas_fake, periods)

        # Step 5: Spearman correlation = Pearson correlation of ranked data
        ic_values = cov / (alphas_std * returns_std)

        del (
            alphas_exp_tensor,
            returns_exp_tensor,
            alphas_mean,
            returns_mean,
            cov,
            alphas_std,
            returns_std,
        )
        torch.cuda.empty_cache()

        return ic_values  # (dates, alphas, periods)

    if not by_group:

        # if not by group, directly compute the ics without masking
        ic_values = compute_ics(alphas_exp, returns_exp)

        # unsqueeze the last dim
        ic_values = ic_values.unsqueeze(dim=-1)

        return ic_values  # (dates, alphas, periods, 1)

    else:

        # if we need to group, first we should create masks by the unique groups
        non_nan_group = groups[~torch.isnan(groups)]
        unique_groups = torch.unique(non_nan_group)

        # expand the group
        # (dates, alphas_fake, periods_fake, stocks)
        group_exp = (
            groups.unsqueeze(1).unsqueeze(1).expand(-1, num_alphas, num_periods, -1)
        )

        groups_lst = []
        # loop over the unqiue groups and create mask one by one
        for group in unique_groups:
            # create a mask where group_exp matches the current unique group
            mask = group_exp == group

            alphas_exp_mask = torch.where(mask, alphas_exp, torch.nan)
            returns_exp_mask = torch.where(mask, returns_exp, torch.nan)

            ic_values_group = compute_ics(alphas_exp_mask, returns_exp_mask)
            groups_lst.append(ic_values_group)

        ic_values_groups = torch.stack(groups_lst, dim=1)

        # unsqueeze the last dim
        ic_values_groups = ic_values_groups.unsqueeze(dim=-1)

        return ic_values_groups  # (dates, groups, alphas, periods, 1)


def information_coefficient_stats(ics: torch.Tensor) -> torch.Tensor:
    """
    Compute the Spearman IC Statistics like:
     mean / positive ratio / std / Risk Adjusted IC / t-stats / p-value / IC Skew / IC Kurtosis

    :param ics: tensor(dates, groups, alphas, periods, 1) if by_group
                tensor(dates, alphas, periods, 1) if not by_group
    :return: tensor(groups, alphas, periods, metrics, 1) if by_group
             tensor(alphas, periods, metrics, 1) if not by_group
    """
    # ------------------------------------
    # squeeze the last dim
    ics = ics.squeeze(dim=-1)

    # ------------------------------------
    # compute the mean
    ics_mean = ics.nanmean(dim=0)

    # ------------------------------------
    # compute the positive ratios
    positive_count = torch.sum(ics > 0, dim=0)
    total_count = ics.size(0)
    positive_ratio = positive_count / total_count

    # ------------------------------------
    # compute the std
    ics_std = nanstd(ics, dim=0)

    # ------------------------------------
    # compute the Risk Adjusted IC
    ics_risk_adjusted = ics_mean / ics_std

    # ------------------------------------
    # compute the t-stats and p-values
    t_stat, p_value = t_test_pytorch(ics, dim=0)

    # ------------------------------------
    # compute the IC Skew
    central = ics - ics.nanmean(dim=0, keepdim=True)
    m3 = (central**3).nanmean(dim=0)
    m2 = (central**2).nanmean(dim=0)
    skew_ = m3 / m2**1.5

    # ------------------------------------
    # compute the IC Kurtosis
    m4 = (central**4).nanmean(dim=0)
    var = nanvar(ics, dim=0)
    kurtosis_ = m4 / var**2 - 3

    # ------------------------------------
    # stack them all on the last dim
    metrics = torch.stack(
        [
            ics_mean,
            positive_ratio,
            ics_std,
            ics_risk_adjusted,
            t_stat,
            p_value,
            skew_,
            kurtosis_,
        ],
        dim=-1,
    )

    return metrics.unsqueeze(dim=-1)


def stratification_stats(return_: torch.Tensor, turnover: torch.Tensor) -> torch.Tensor:
    """
    compute the statistics of the stratified returns

    :param return_: tensor(dates, alphas, groups, layers, periods, 1) if by_group
                    tensor(dates, alphas, layers, periods, 1) if not by_group
           turnover: tensor(alphas, groups, layers, periods, 1) if by_group
                     tensor(alphas, layers, periods, 1) if not by_group

    :return: tensor(alphas, groups, layers, periods, metrics, 1) if by_group
             tensor(alphas, layers, periods, 1) if not by_group
    """

    # ------------------------------------
    # squeeze the last dim
    return_ = return_.squeeze(dim=-1)
    turnover = turnover.squeeze(dim=-1)

    # ------------------------------------
    # compute the mean
    ret_mean = return_.nanmean(dim=0)

    # ------------------------------------
    # compute the positive ratios
    positive_count = torch.sum(return_ > 0, dim=0)
    total_count = return_.size(0)
    win_rate = positive_count / total_count

    # ------------------------------------
    # compute the std
    ret_std = nanstd(return_, dim=0)

    # ------------------------------------
    # compute the Risk Adjusted IC
    ret_adjusted = ret_mean / ret_std

    # ------------------------------------
    # compute the t-stats and p-values
    t_stat, p_value = t_test_pytorch(return_, dim=0)

    # ------------------------------------
    # stack them all on the last dim
    ret_metrics = torch.stack(
        [ret_mean, win_rate, ret_std, ret_adjusted, t_stat, p_value, turnover],
        dim=-1,
    )

    return ret_metrics.unsqueeze(dim=-1)
    # if by_group: (alphas, groups, layers, periods, n_metrics, 1)
    # else: (alphas, layers, periods, n_metrics, 1)


def long_short_backtest(
    alphas: torch.Tensor,
    returns: torch.Tensor,
    metrics: torch.Tensor,
    backtest_periods: Tuple,
    groups: Optional[torch.Tensor] = None,
    by_group: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
    """
    :param alphas: (dates, alphas, stocks)
    :param returns: (dates, periods, stocks) The one-period return should be sliced before using this func
    :param metrics: tensor(groups, alphas, periods, metrics) if by_group
                   tensor(alphas, periods, metrics) if not by_group
    :param backtest_periods: a tuple specifying the backtesting periods, it's a good idea just to use the backtest periods for ic computing
    :param groups: (dates, stocks)
    :param by_group: decide if we do the long short backtest by groups
    :return:
    """
    # ------------------------------------
    # record the important shapes
    n_dates, n_alphas, n_stocks = alphas.shape

    # ------------------------------------
    # only keep the daily return
    returns = returns[:, 0, :]
    returns = returns.unsqueeze(1).expand(-1, n_alphas, -1)
    metrics = metrics.squeeze(-1)

    # ------------------------------------
    # check if by_group and groups are consistent / if the mode parameter is within the options
    if by_group and (groups is None):
        raise ValueError("please provide the groups tensor")

    # ------------------------------------
    # deal the metrics
    if not by_group:
        metric = metrics[:, :, 0]  # (alphas, periods)
        metric = metric.nanmean(dim=-1)  # (alphas)

    else:
        metric = metrics[:, :, :, 0]  # (groups, alphas, periods)
        metric = metric.nanmean(dim=-1)  # (groups, alphas)

    if not by_group:

        # step 1: initialization
        modes_return_lst = []
        modes_turnover_lst = []
        metric = (
            metric.unsqueeze(dim=0).unsqueeze(dim=2).expand(n_dates, n_alphas, n_stocks)
        )  # (dates, alphas, stocks)

        # step 2: loop over the modes
        # todo: now only do the long short
        for mode in [BacktestModes.LONG, BacktestModes.SHORT, BacktestModes.LONG_SHORT]:
            # for mode in [BacktestModes.LONG_SHORT]:
            # step 3: loop over the periods
            periods_return_lst = []
            periods_turnover_lst = []
            for backtest_period in backtest_periods:
                tensor_return, tensor_turnover = longshort_alpha_return(
                    alphas=alphas,
                    return_=returns,
                    metric=metric,
                    mode=mode,
                    n_dates=n_dates,
                    backtest_period=backtest_period,
                )  # (dates, alphas)

                periods_return_lst.append(tensor_return)
                periods_turnover_lst.append(tensor_turnover)

            tensor_return_periods = torch.stack(
                periods_return_lst, dim=-1
            )  # (dates, alphas, periods)
            tensor_turnover_periods = torch.stack(
                periods_turnover_lst, dim=-1
            )  # (alphas, periods)
            modes_return_lst.append(tensor_return_periods)
            modes_turnover_lst.append(tensor_turnover_periods)

        # step 4: stack the general tensor
        tensor_modes_periods_return = torch.stack(
            modes_return_lst, dim=2
        )  # (dates, alphas, modes, periods)
        tensor_modes_periods_turnover = torch.stack(
            modes_turnover_lst, dim=1
        )  # (alphas, modes, periods)

        return tensor_modes_periods_return.unsqueeze(
            dim=-1
        ), tensor_modes_periods_turnover.unsqueeze(dim=-1)
        # (dates, alphas, modes, periods, 1), (alphas, modes, periods, 1)

    else:

        # step 1: initialization / get all the unique groups
        non_nan_group = groups[~torch.isnan(groups)]
        unique_groups = torch.unique(non_nan_group)
        groups = groups.unsqueeze(dim=1).expand(
            n_dates, n_alphas, n_stocks
        )  # (dates, alphas, stocks)

        groups_return_lst = []
        groups_turnover_lst = []

        # step 2: loop over all the groups
        for group in unique_groups:

            modes_return_lst = []
            modes_turnover_lst = []

            metric_group = metric[int(group.item()), :]
            metric_group = (
                metric_group.unsqueeze(dim=0)
                .unsqueeze(dim=2)
                .expand(n_dates, n_alphas, n_stocks)
            )
            # (dates, alphas, stocks)
            # create a mask where groups matches the current unique group
            mask = groups == group

            # deselect the corresponding alphas and returns
            alphas_group = torch.where(mask, alphas, torch.nan)
            returns_group = torch.where(mask, returns, torch.nan)
            metric_group = torch.where(mask, metric_group, torch.nan)

            # step 3: loop over the modes
            # todo: now only do the long short
            for mode in [
                BacktestModes.LONG,
                BacktestModes.SHORT,
                BacktestModes.LONG_SHORT,
            ]:
                # for mode in [BacktestModes.LONG_SHORT]:
                # step 4: loop over the periods
                periods_return_lst = []
                periods_turnover_lst = []
                for backtest_period in backtest_periods:
                    tensor_return, tensor_turnover = longshort_alpha_return(
                        alphas=alphas_group,
                        return_=returns_group,
                        metric=metric_group,
                        mode=mode,
                        n_dates=n_dates,
                        backtest_period=backtest_period,
                    )  # (dates, alphas)

                    periods_return_lst.append(tensor_return)
                    periods_turnover_lst.append(tensor_turnover)

                tensor_return_periods = torch.stack(
                    periods_return_lst, dim=2
                )  # (dates, alphas, periods)
                tensor_turnover_periods = torch.stack(
                    periods_turnover_lst, dim=-1
                )  # (alphas, periods)
                modes_return_lst.append(tensor_return_periods)
                modes_turnover_lst.append(tensor_turnover_periods)

            # step 5: stack the general tensor
            tensor_modes_periods_return = torch.stack(
                modes_return_lst, dim=2
            )  # (dates, alphas, modes, periods)
            tensor_modes_periods_turnover = torch.stack(
                modes_turnover_lst, dim=1
            )  # (alphas, modes, periods)
            groups_return_lst.append(tensor_modes_periods_return)
            groups_turnover_lst.append(tensor_modes_periods_turnover)

        tensor_groups_modes_periods_return = torch.stack(
            groups_return_lst, dim=2
        )  # (dates, alphas, groups, modes, periods)
        tensor_groups_modes_periods_turnover = torch.stack(
            groups_turnover_lst, dim=1
        )  # (alphas, groups, modes, periods)

        return tensor_groups_modes_periods_return.unsqueeze(
            dim=-1
        ), tensor_groups_modes_periods_turnover.unsqueeze(dim=-1)
        # (dates, alphas, groups, modes, periods, 1), (alphas, groups, modes, periods, 1)


def stratified_backtest(
    alphas: torch.Tensor,
    returns: torch.Tensor,
    metrics: torch.Tensor,
    backtest_periods: Tuple,
    n_stratify: int,
    groups: Optional[torch.Tensor] = None,
    by_group: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
    """
    :param alphas: (dates, alphas, stocks)
    :param returns: (dates, periods, stocks) The one-period return should be sliced before using this func
    :param metrics: tensor(groups, alphas, periods, metrics) if by_group
                   tensor(alphas, periods, metrics) if not by_group
    :param backtest_periods: a tuple specifying the backtesting periods, it's a good idea just to use the backtest periods for ic computing
    :param groups: (dates, stocks)
    :param by_group: decide if we do the long short backtest by groups
    :return:
    """
    # ------------------------------------
    # record the important shapes
    n_dates, n_alphas, n_stocks = alphas.shape

    # ------------------------------------
    # only keep the daily return
    returns = returns[:, 0, :]
    returns = returns.unsqueeze(1).expand(-1, n_alphas, -1)
    metrics = metrics.squeeze(-1)

    # ------------------------------------
    # check if by_group and groups are consistent / if the mode parameter is within the options
    if by_group and (groups is None):
        raise ValueError("please provide the groups tensor")

    # ------------------------------------
    # deal the metrics
    if not by_group:
        metric = metrics[:, :, 0]  # (alphas, periods)
        metric = metric.nanmean(dim=-1)  # (alphas)
    else:
        metric = metrics[:, :, :, 0]  # (groups, alphas, periods)
        metric = metric.nanmean(dim=-1)  # (groups, alphas)

    if not by_group:

        # step 1: initialization
        stratify_return_lst = []
        stratify_turnover_lst = []
        metric = (
            metric.unsqueeze(dim=0).unsqueeze(dim=2).expand(n_dates, n_alphas, n_stocks)
        )  # (dates, alphas, stocks)

        # step 2: loop over the layers
        # todo: now only do the long short
        for layer in range(n_stratify):
            # step 3: loop over each layer in the stratification
            periods_return_lst = []
            periods_turnover_lst = []
            for backtest_period in backtest_periods:
                tensor_return, tensor_turnover = stratified_alpha_return(
                    alphas=alphas,
                    return_=returns,
                    metric=metric,
                    layer=layer,
                    n_stratify=n_stratify,
                    n_dates=n_dates,
                    backtest_period=backtest_period,
                )  # (dates, alphas)

                periods_return_lst.append(tensor_return)
                periods_turnover_lst.append(tensor_turnover)

            tensor_return_periods = torch.stack(
                periods_return_lst, dim=-1
            )  # (dates, alphas, periods)
            tensor_turnover_periods = torch.stack(
                periods_turnover_lst, dim=-1
            )  # (alphas, periods)
            stratify_return_lst.append(tensor_return_periods)
            stratify_turnover_lst.append(tensor_turnover_periods)

        # step 4: stack the general tensor
        tensor_layers_periods_return = torch.stack(
            stratify_return_lst, dim=2
        )  # (dates, alphas, layers, periods)
        tensor_layers_periods_turnover = torch.stack(
            stratify_turnover_lst, dim=1
        )  # (alphas, layers, periods)

        return tensor_layers_periods_return.unsqueeze(
            dim=-1
        ), tensor_layers_periods_turnover.unsqueeze(dim=-1)
        # (dates, alphas, layers, periods, 1), (alphas, layers, periods, 1)

    else:

        # step 1: initialization / get all the unique groups
        non_nan_group = groups[~torch.isnan(groups)]
        unique_groups = torch.unique(non_nan_group)
        groups = groups.unsqueeze(dim=1).expand(
            n_dates, n_alphas, n_stocks
        )  # (dates, alphas, stocks)

        groups_return_lst = []
        groups_turnover_lst = []

        # step 2: loop over all the groups
        for group in unique_groups:

            stratify_return_lst = []
            stratify_turnover_lst = []

            metric_group = metric[int(group.item()), :]
            metric_group = (
                metric_group.unsqueeze(dim=0)
                .unsqueeze(dim=2)
                .expand(n_dates, n_alphas, n_stocks)
            )
            # (dates, alphas, stocks)
            # create a mask where groups matches the current unique group
            mask = groups == group

            # deselect the corresponding alphas and returns
            alphas_group = torch.where(mask, alphas, torch.nan)
            returns_group = torch.where(mask, returns, torch.nan)
            metric_group = torch.where(mask, metric_group, torch.nan)

            # step 3: loop over the layers
            # todo: now only do the long short
            for layer in range(n_stratify):
                # step 4: loop over the periods
                periods_return_lst = []
                periods_turnover_lst = []
                for backtest_period in backtest_periods:
                    tensor_return, tensor_turnover = stratified_alpha_return(
                        alphas=alphas_group,
                        return_=returns_group,
                        metric=metric_group,
                        layer=layer,
                        n_stratify=n_stratify,
                        n_dates=n_dates,
                        backtest_period=backtest_period,
                    )  # (dates, alphas)

                    periods_return_lst.append(tensor_return)
                    periods_turnover_lst.append(tensor_turnover)

                tensor_return_periods = torch.stack(
                    periods_return_lst, dim=2
                )  # (dates, alphas, periods)
                tensor_turnover_periods = torch.stack(
                    periods_turnover_lst, dim=-1
                )  # (alphas, periods)
                stratify_return_lst.append(tensor_return_periods)
                stratify_turnover_lst.append(tensor_turnover_periods)

            # step 5: stack the general tensor
            tensor_layers_periods_return = torch.stack(
                stratify_return_lst, dim=2
            )  # (dates, alphas, layers, periods)
            tensor_layers_periods_turnover = torch.stack(
                stratify_turnover_lst, dim=1
            )  # (alphas, layers, periods)
            groups_return_lst.append(tensor_layers_periods_return)
            groups_turnover_lst.append(tensor_layers_periods_turnover)

        tensor_groups_layers_periods = torch.stack(
            groups_return_lst, dim=2
        )  # (dates, alphas, groups, modes, periods)
        tensor_groups_layers_turnover = torch.stack(
            groups_turnover_lst, dim=1
        )  # (alphas, groups, modes, periods)

        return tensor_groups_layers_periods.unsqueeze(
            dim=-1
        ), tensor_groups_layers_turnover.unsqueeze(dim=-1)

        # (dates, alphas, groups, layers, periods, 1), (alphas, groups, layers, periods, 1)


def longshort_alpha_return(
    alphas: torch.Tensor,
    return_: torch.Tensor,
    metric: torch.Tensor,
    mode: BacktestModes,
    backtest_period: int,
    n_dates: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param alphas: (dates, alphas, stocks)
    :param return_: (dates, stocks) the one-day return
    :param metric: (alphas) the average ic over the specified periods for a certain alpha for a certain group which should be averaged outside this func
    :param mode: ['long', 'short', 'long_short']
    :param backtest_period: int
    :return: the returns of each alpha on each day, the turnover of each alpha, averaged over the dates
    """
    # todo: add long and short
    assert mode in [
        BacktestModes.LONG,
        BacktestModes.SHORT,
        BacktestModes.LONG_SHORT,
    ], "please specify the mode parameter as one of [BacktestMode.LONG, BacktestMode.SHORT, BacktestMode.LONG_SHORT]"

    # Step 1: change the sign of the alphas according to the sign of ics
    ic_sign = torch.sign(metric)
    ic_sign = torch.where(ic_sign == 0, 1.0, ic_sign)
    alphas = alphas * ic_sign
    nan_mask = torch.isnan(alphas)

    # Step 2: get the long / short mask respectively, and assign the weights
    mask_long = (
        alphas >= torch.nanquantile(alphas, q=long_lower, dim=-1, keepdim=True)
    ) & (alphas <= torch.nanquantile(alphas, q=long_upper, dim=-1, keepdim=True))

    mask_short = (
        alphas >= torch.nanquantile(alphas, q=short_lower, dim=-1, keepdim=True)
    ) & (alphas <= torch.nanquantile(alphas, q=short_upper, dim=-1, keepdim=True))

    alphas = torch.zeros_like(alphas)  # Start with all zeros
    alphas = torch.where(mask_long, 1, alphas)  # Set long positions to 1
    alphas = torch.where(mask_short, -1, alphas)  # Set short positions to -1

    # Step 3: keep the nan mask
    alphas = torch.where(nan_mask, torch.nan, alphas)  # (dates, alphas, stocks)

    # Step 4: deal with the backtesting mode (long / short / long-short)
    if mode == BacktestModes.LONG:
        long_mask = alphas > 0
        alphas = torch.where(long_mask, alphas, torch.nan)
    elif mode == BacktestModes.SHORT:
        short_mask = alphas < 0
        alphas = torch.where(short_mask, alphas, torch.nan)
    else:
        pass

    # Step 5: compute the weights
    # (dates, alphas, stocks)
    weights = torch.nan_to_num(
        alphas / torch.nansum(alphas.abs(), dim=-1, keepdim=True), nan=0.0
    ) * (
        1 / backtest_period
    )  # (dates, alphas, stocks)

    # Step 6: deal with the backtest period, if the period is larger than 1, we smooth it
    # first, create an empty tensor filled with 0s to accumulate all the returns
    multiple_returns = torch.zeros(
        weights.shape[0], weights.shape[1], device=weights.device
    )
    turnovers = torch.zeros(weights.shape[1], device=weights.device)

    for period_ in torch.arange(backtest_period):
        # (dates / backtest_period - period_, alphas, stocks)
        weights_temp = weights[period_::backtest_period, :, :]

        # compute the turnover rate
        # compute the single-sided turnover rate first
        first = weights_temp[:-1, :, :]
        second = weights_temp[1:, :, :]

        # get the turnover series which will be used for computing the commission fee
        turnover_series = (second - first).abs().nansum(
            dim=-1
        ) / 2  # (dates / backtest_period - period_, alphas)
        buff_turnover = torch.full(
            (period_, *turnover_series.shape[1:]), 0.0, device=turnover_series.device
        )
        first_turnover = (
            weights_temp[0, :].abs().nansum(dim=-1).unsqueeze(0)
        )  # (1, alphas)
        turnover_slice = torch.cat([first_turnover, turnover_series], dim=0)

        turnover_temp = torch.zeros(
            (backtest_period * turnover_slice.shape[0], *turnover_slice.shape[1:]),
            device=turnover_slice.device,
        )
        turnover_temp[::backtest_period, :] = turnover_slice
        turnover_temp = torch.cat([buff_turnover, turnover_temp], dim=0)
        turnover_temp = turnover_temp[:n_dates, :]

        turnover = turnover_series.nanmean(dim=0)

        buff_ = torch.full(
            (period_, *weights.shape[1:]), torch.nan, device=weights.device
        )
        weights_temp = weights_temp.repeat_interleave(backtest_period, dim=0)
        weights_temp = torch.cat((buff_, weights_temp), dim=0)
        weights_temp = weights_temp[:n_dates, :, :]

        # Step 7: compute the daily returns
        single_return = (weights_temp * return_).nansum(dim=-1) - float(
            commission
        ) * turnover_temp * 2  # (dates, alphas)

        # Step 8: compute the daily returns
        multiple_returns += single_return
        turnovers += turnover

    return multiple_returns, turnovers


def stratified_alpha_return(
    alphas: torch.Tensor,
    return_: torch.Tensor,
    metric: torch.Tensor,
    layer: int,
    n_stratify: int,
    backtest_period: int,
    n_dates: int,
    mask_group: Union[torch.Tensor, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param alphas: (dates, alphas, stocks)
    :param return_: (dates, alphas, stocks) the one-day return
    :param metric: (alphas) the average ic over the specified periods for a certain alpha for a certain group which should be averaged outside this func
    :param layer: the specific layer of the n_stratify here to backtest
    :param n_stratify: the total stratifications we wanna backtest
    :param backtest_period: int
    :return: the returns of each alpha on each day, the turnover of each alpha, averaged over the dates
    """

    # Step 0: compute the market ret, clip the return_ at the 5% and 95% percentile
    # Compute quantiles for clipping
    lower_bound = torch.nanquantile(return_, 0.05, dim=-1, keepdim=True)
    upper_bound = torch.nanquantile(return_, 0.95, dim=-1, keepdim=True)

    # Clip using torch.where
    return_mkt = torch.where(return_ < lower_bound, lower_bound, return_)
    return_mkt = torch.where(return_mkt > upper_bound, upper_bound, return_mkt)

    return_mkt = return_mkt.nanmean(dim=-1)  # (dates, alphas)

    # deselect the corresponding alphas and returns according to the mask_group
    if mask_group is not None:
        alphas = torch.where(mask_group, alphas, torch.nan)
        return_ = torch.where(mask_group, return_, torch.nan)
        metric = torch.where(mask_group, metric, torch.nan)
    else:
        pass

    # Step 1: change the sign of the alphas according to the sign of ics
    ic_sign = torch.sign(metric)
    ic_sign = torch.where(ic_sign == 0, 1.0, ic_sign)
    alphas = alphas * ic_sign
    nan_mask = torch.isnan(alphas)

    # Step 2: generate the layer mask according to layer and n_stratify
    lower = layer / n_stratify
    upper = (layer + 1) / n_stratify
    alphas = (alphas >= torch.nanquantile(alphas, lower, dim=-1, keepdim=True)) & (
        alphas <= torch.nanquantile(alphas, upper, dim=-1, keepdim=True)
    )

    # Step 3: keep the nan mask
    alphas = torch.where(nan_mask, torch.nan, alphas)  # (dates, alphas, stocks)

    # Step 4: compute the weights
    # (dates, alphas, stocks)
    weights = torch.nan_to_num(
        alphas / torch.nansum(alphas.abs(), dim=-1, keepdim=True), nan=0.0
    ) * (
        1 / backtest_period
    )  # (dates, alphas, stocks)

    # Step 5: deal with the backtest period, if the period is larger than 1, we smooth it
    # first, create an empty tensor filled with 0s to accumulate all the returns
    multiple_returns = torch.zeros(
        weights.shape[0], weights.shape[1], device=weights.device
    )
    turnovers = torch.zeros(weights.shape[1], device=weights.device)

    for period_ in torch.arange(backtest_period):
        # (dates / backtest_period - period_, alphas, stocks)
        weights_temp = weights[period_::backtest_period, :, :]

        # compute the turnover rate
        # compute the single-sided turnover rate first
        first = weights_temp[:-1, :, :]
        second = weights_temp[1:, :, :]

        # get the turnover series which will be used for computing the commission fee
        turnover_series = (second - first).abs().nansum(
            dim=-1
        ) / 2  # (dates / backtest_period - period_, alphas)
        buff_turnover = torch.full(
            (period_, *turnover_series.shape[1:]), 0.0, device=turnover_series.device
        )
        first_turnover = (
            weights_temp[0, :].abs().nansum(dim=-1).unsqueeze(0)
        )  # (1, alphas)
        turnover_slice = torch.cat([first_turnover, turnover_series], dim=0)

        turnover_temp = torch.zeros(
            (backtest_period * turnover_slice.shape[0], *turnover_slice.shape[1:]),
            device=turnover_slice.device,
        )
        turnover_temp[::backtest_period, :] = turnover_slice
        turnover_temp = torch.cat([buff_turnover, turnover_temp], dim=0)
        turnover_temp = turnover_temp[:n_dates, :]

        turnover = turnover_series.nanmean(dim=0)

        buff_ = torch.full(
            (period_, *weights.shape[1:]), torch.nan, device=weights.device
        )
        weights_temp = weights_temp.repeat_interleave(backtest_period, dim=0)
        weights_temp = torch.cat((buff_, weights_temp), dim=0)
        weights_temp = weights_temp[:n_dates, :, :]

        # Step 7: compute the daily returns
        single_return = (weights_temp * return_).nansum(
            dim=-1
        ) - commission * turnover_temp * 2  # (dates, 1)

        # Step 8: compute the daily returns
        multiple_returns += single_return
        turnovers += turnover

    multiple_returns = multiple_returns - return_mkt

    return multiple_returns, turnovers


def return2netvalue(returns: torch.Tensor):
    """';
    :param returns:
    :return:
    """
    returns = returns + 1.0

    return torch.cumprod(returns, dim=0)


def ic2cumsum(ic: torch.Tensor):
    """';
    :param ic:
    :return:
    """

    return torch.cumsum(ic, dim=0)


def return2drawdown(returns: torch.Tensor):
    """
    :param returns: (dates, groups, combos, periods, 1) if groupby else (dates, groups, periods, 1)
    :return:
    """
    returns = returns + 1.0
    portfolio_values = torch.cumprod(returns, dim=0)
    running_max = torch.cummax(portfolio_values, dim=0)[0]
    drawdowns = (portfolio_values - running_max) / running_max

    return drawdowns

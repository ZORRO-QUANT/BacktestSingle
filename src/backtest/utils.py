import datetime
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Literal, List, Union

import numpy as np
import pandas as pd
import paramiko
import torch
from scipy.stats import t

from functools import lru_cache
from pathlib import Path
import yaml
from typing import Dict, Any

# Constants should be uppercase
PROJECT_ROOT = Path(__file__).parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@lru_cache(maxsize=None)
def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_file: Name of the YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = PROJECT_ROOT / "config" / config_file
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML parsing error: {e}")


def sublist(whole_list: List, chunk_size: int = 30):
    for i in range(0, len(whole_list), chunk_size):
        yield whole_list[i : i + chunk_size]


def nanstd(tensor: torch.Tensor, dim: int, keepdim: bool = False):
    """Compute the standard deviation manually while ignoring NaNs."""
    # Mask NaN values
    valid_mask = ~torch.isnan(tensor)
    count = valid_mask.sum(dim=dim, keepdim=True).float()

    # Compute the mean, while ignoring NaNs (ensure broadcasting along the correct dimension)
    mean = torch.nansum(tensor, dim=dim, keepdim=True) / count.clamp(min=1)

    # Compute variance: sum of squared differences from the mean, divided by (count - 1)
    variance = torch.nansum((tensor - mean) ** 2, dim=dim, keepdim=True) / count.clamp(
        min=1
    )

    # Standard deviation is the square root of variance
    std = torch.sqrt(variance)

    if not keepdim:
        std = std.squeeze(dim=dim)
        count = count.squeeze(dim=dim)

    # replace the place where the whole dim are nans
    std[count == 0] = torch.nan

    return std


def nanvar(tensor: torch.Tensor, dim: int, keepdim: bool = False):
    """Compute the standard deviation manually while ignoring NaNs."""
    # Mask NaN values
    valid_mask = ~torch.isnan(tensor)
    count = valid_mask.sum(dim=dim, keepdim=True).float()

    # Compute the mean, while ignoring NaNs (ensure broadcasting along the correct dimension)
    mean = torch.nansum(tensor, dim=dim, keepdim=True) / count.clamp(min=1)

    # Compute variance: sum of squared differences from the mean, divided by (count - 1)
    variance = torch.nansum((tensor - mean) ** 2, dim=dim, keepdim=True) / count.clamp(
        min=1
    )

    if not keepdim:
        variance = variance.squeeze(dim=dim)
        count = count.squeeze(dim=dim)

    # replace the place where the whole dim are nans
    variance[count == 0] = torch.nan

    return variance


def t_test_pytorch(tensor: torch.Tensor, dim: int, population_mean: float = 0):
    # Sample mean along the specified dimension
    sample_mean = tensor.nanmean(dim=dim)

    # Sample standard deviation along the specified dimension (ignoring NaNs)
    sample_std = nanstd(tensor, dim=dim)

    # Number of non-NaN elements along the specified dimension
    sample_size = (~torch.isnan(tensor)).sum(dim=dim).float()

    # Compute t-statistic
    t_stat = (sample_mean - population_mean) / (sample_std / torch.sqrt(sample_size))

    # Calculate p-value from the t-statistic using SciPy
    # The p-value is calculated for a two-tailed test
    p_value = (
        2
        * (
            1
            - torch.tensor(t.cdf(t_stat.abs().cpu().numpy(), df=sample_size.cpu() - 1))
        )
    ).to(device=tensor.device, dtype=tensor.dtype)

    return t_stat, p_value


def get_empty_df(
    start: datetime.datetime,
    end: datetime.datetime,
    frequency: str,
    inclusive: Literal["left", "right", "both", "neither"],
):
    """
    generate the continuous empty dataframe to deal with the missing data
    """
    open_list = pd.date_range(start=start, end=end, freq=frequency, inclusive=inclusive)

    df_empty = pd.DataFrame({"time": open_list})
    df_empty["time"] = pd.to_datetime(df_empty["time"])
    df_empty["closeTime"] = (
        df_empty["time"] + pd.Timedelta(frequency) - pd.Timedelta(seconds=1)
    )

    return df_empty


def tictoc(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        func(*args, **kwargs)
        t2 = time.time() - t1
        print(rf"{func.__name__}花费 {t2} 秒")

    return wrapper


def upload_dir(sftp, local_dir: Path, remote_dir: Path):
    """Recursively upload a local directory to a remote directory."""
    # Ensure the local directory exists. If using pathlib, no need to use os.makedirs.

    # Ensure the remote directory exists, or create it
    current_path = Path("/")
    for part in remote_dir.parts:
        current_path = current_path / part
        try:
            sftp.mkdir(str(current_path))
        except IOError:  # Directory already exists
            pass

    for item in local_dir.iterdir():
        local_path = item
        remote_path = remote_dir / item.name

        if local_path.is_file():
            sftp.put(str(local_path), str(remote_path))
        else:  # it's a directory
            try:
                sftp.mkdir(str(remote_path))
            except IOError:
                pass  # Ignore if remote directory already exists
            upload_dir(sftp, local_path, remote_path)


def send_dir_to_windows(
    remote_dir: Path,
    local_dir: Path,
    windows_username: str,
    windows_ip: str,
    windows_password: str,
):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            hostname=windows_ip, username=windows_username, password=windows_password
        )
        sftp = client.open_sftp()
        upload_dir(sftp, remote_dir, local_dir)
        logging.info("Directory transfer completed successfully.")
        sftp.close()

        logging.info(f"File {remote_dir} uploaded successfully to {local_dir}")
    except Exception as e:
        logging.warning(f"An error occurred: {e}")
    finally:
        client.close()


def upload_file(sftp, local_file: Path, remote_file: Path):
    """Upload a single local file to a specific remote file path."""
    # Ensure the local file is indeed a file
    if not local_file.is_file():
        raise ValueError(
            f"{local_file} is not a file. Only individual files are allowed."
        )

    # Ensure the remote directory exists, or create it
    remote_dir = remote_file.parent
    current_path = Path("/")
    for part in remote_dir.parts:
        current_path = current_path / part
        try:
            sftp.mkdir(str(current_path))
        except IOError:  # Directory already exists
            pass

    # Upload the file to the specified remote path
    sftp.put(str(local_file), str(remote_file))
    logging.info(f"File {local_file} uploaded successfully to {remote_file}")


def send_file_to_windows(
    remote_path: Path,
    local_path: Path,
    windows_username: str,
    windows_ip: str,
    windows_password: str,
):
    """Send a single file from a local path to a specific remote path on a Windows machine."""

    # Verify that the local path is indeed a file
    if not remote_path.is_file():
        raise ValueError(
            f"{remote_path} is not a file. Only individual files are allowed."
        )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            hostname=windows_ip, username=windows_username, password=windows_password
        )
        sftp = client.open_sftp()
        upload_file(sftp, remote_path, local_path)
        sftp.close()
        print("File transfer completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close()


def compute_forward_returns(
    factor, prices, periods=(1, 5, 10), filter_zscore=None, cumulative_returns=True
):
    """
    Finds the N period forward returns (as percent change) for each asset
    provided.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.

        - See full explanation in utils.get_clean_factor_and_forward_returns

    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    cumulative_returns : bool, optional
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the forward returns for assets.
        Forward returns column names follow the format accepted by
        pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc).
        'date' index freq property (forward_returns.index.levels[0].freq)
        will be set to a trading calendar (pandas DateOffset) inferred
        from the input data (see infer_trading_calendar for more details).
    """

    factor_dateindex = factor.index.levels[0]
    factor_dateindex = factor_dateindex.intersection(prices.index)

    if len(factor_dateindex) == 0:
        raise ValueError(
            "Factor and prices indices don't match: make sure "
            "they have the same convention in terms of datetimes "
            "and symbol-names"
        )

    # chop prices down to only the assets we care about (= unique assets in
    # `factor`).  we could modify `prices` in place, but that might confuse
    # the caller.
    prices = prices.filter(items=factor.index.levels[1])

    raw_values_dict = {}
    column_list = []

    for period in sorted(periods):
        if cumulative_returns:
            returns = prices.pct_change(period)
        else:
            returns = prices.pct_change()

        forward_returns = returns.shift(-period).reindex(factor_dateindex)

        if filter_zscore is not None:
            mask = abs(forward_returns - forward_returns.mean()) > (
                filter_zscore * forward_returns.std()
            )
            forward_returns[mask] = np.nan

        label = rf"{period}"

        column_list.append(label)

        raw_values_dict[label] = np.concatenate(forward_returns.values)

    df = pd.DataFrame.from_dict(raw_values_dict)
    df.set_index(
        pd.MultiIndex.from_product(
            [factor_dateindex, prices.columns], names=["time", "symbol"]
        ),
        inplace=True,
    )

    # now set the columns correctly
    df = df[column_list]

    # df.index.levels[0].freq = freq
    df.index.set_names(["time", "symbol"], inplace=True)

    return df


def save_to_excel(
    path: Union[Path, str], df: pd.DataFrame, sheet_name: str, index: bool = False
):
    """Smart engine selection based on file existence"""
    if not os.path.exists(path):
        # xlsxwriter for new files (fastest)
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)
    else:
        # openpyxl for appending (slower but necessary)
        with pd.ExcelWriter(
            path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)

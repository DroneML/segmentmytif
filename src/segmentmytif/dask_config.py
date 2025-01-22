from typing import Literal
from dask.distributed import LocalCluster


def get_dask_config(mode: Literal["normal", "safe", "parallel"] = "normal", **lckwargs):
    """Get configuration strings for dask.

    This function translates the settings of dask.config to users.
    "normal" will be the default configuration "threads", which is for multi-threaded.
    "safe" will be the configuration "synchronous", which is for single-threaded.
    "parallel" will


    Parameters
    ----------
    mode : Literal["normal", "safe", "parallel"], optional
        Choice of dask configuration, by default "normal"
    lckwargs : dict
        Keyword arguments for LocalCluster
    """
    match mode:
        case "normal":
            return "threads"
        case "safe":
            return "synchronous"
        case "parallel":
            return LocalCluster(lckwargs)
        case _:
            msg = f"Unknown mode: {mode}"
            raise ValueError(msg)

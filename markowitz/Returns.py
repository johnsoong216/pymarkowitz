"""
ReturnGenerator generates return matrices used in mean-variance optimization problems.
Daily, Rolling, Interval Return as well as Arithmetic/Geometric Mean Return can be generated using this module
"""

import numpy as np
import pandas as pd
from .Exceptions import *


class ReturnGenerator:

    def __init__(self, price_data, assets=None):

        """
        Initialize a ReturnGenerator instance with price data
        :param price_data: pd.DataFrame/np.ndarray, price data of assets
        :param assets: List[str], optional, if passed in an np.ndarray, will use this parameter to record asset names
        """

        if isinstance(price_data, pd.DataFrame):
            self.price_mat = price_data.values.T
            self.assets = price_data.columns
            self.index = price_data.index

        elif isinstance(price_data, np.ndarray):

            self.price_mat = price_data
            self.assets = assets
            self.index = np.arange(0, len(price_data), 1)
        else:
            raise FormatException("""Invalid Format. Price data must be passed in as np.ndarray or pd.DataFrame""")

    def calc_return(self, method, ret_format='df', **kwargs):

        """
        Calculates return based on user-defined method and parameters
        :param method: str, options = ["daily", "rolling", "collapse"]
                    daily: Calculates daily percentage change
                    rolling: Calculates rolling percentage change based on window, user passes in a parameter window=?
                    collapse: Calculates percentage change based on window interval, user passes in a parameter window=?
                              Example: If user passes in window=22, then the output is the return between each 22 day interval from the beginnning

                    additional option: Calculates continuous return by passing in log=True, or discrete return by log=False
        :param ret_format: str, default='df', returns the return matrix in pd.DataFrame or a tuple of asset names and np.ndarray
        :param kwargs: arguments passed into method return_formula()
        """

        price_mat = self.price_mat
        index = self.index

        if method == 'daily':
            ret_mat, ret_idx = ReturnGenerator.return_formula(price_mat, index, window=1, roll=True, **kwargs)
        elif method == 'rolling':
            ret_mat, ret_idx = ReturnGenerator.return_formula(price_mat, index, roll=True, **kwargs)
        elif method == 'collapse':
            ret_mat, ret_idx = ReturnGenerator.return_formula(price_mat, index, roll=False, **kwargs)
        else:
            raise MethodException("""Invalid Method. Valid Inputs: daily, rolling, collapse""")

        return_df = pd.DataFrame(ret_mat.T, columns=self.assets, index=ret_idx)

        if ret_format == 'df':
            return return_df
        elif ret_format == 'raw':
            return self.assets, ret_idx, ret_mat
        else:
            raise FormatException("Invalid Format. Valid options are: df, raw")

    def calc_mean_return(self, method, time_scaling=252, ret_format='series', **kwargs):
        """
        Calculates mean historical return to be used in mean-variance optimizer
        :param method: str, options=["arithmetic", "geometric"]
                    Arithmetic: Calculates the arithmetic mean of return, all paramters in calc_return method can be passed in as additional arguments
                    Geometric: Calculates the geometric mean from first to last observation
        :param time_scaling: int, default=252, annualizes daily mean return
        :param ret_format: pd.Series/np.ndarray, returns a pd.Series object of mean return or a tuple of asset names and np.ndarray
        :param kwargs: additional arguments if using arithmetic
        """

        price_mat = self.price_mat
        index = self.index

        ret_mat, ret_idx = ReturnGenerator.return_formula(price_mat, index, **kwargs)

        if method == 'arithmetic':
            mean_ret = np.mean(ret_mat, axis=1) * time_scaling
        elif method == 'geometric':
            mean_ret = (price_mat[:, -1]/price_mat[:, 0]) ** (1/price_mat.shape[0]) ** time_scaling - 1
        else:
            raise MethodException("""Method options are arithmetic/geometric""")

        if ret_format == 'series':
            return pd.Series(mean_ret, index=self.assets)
        elif ret_format == "raw":
            return self.assets, mean_ret
        else:
            raise FormatException("""Invalid Format. Valid options are: series, raw""")

    @staticmethod
    def return_formula(price_mat, index, roll=True, window=1, log=False):
        """
        Converts price data into return data
        :param price_mat: np.ndarray, price matrix
        :param index: list/index, dates/observation time scale
        :param roll: bool, rolling/collapse return
        :param window: int, time interval
        :param log: bool, continuous/discrete
        :return:
        """

        if roll:
            step = 1
            shift = window
        else:
            shift = window
            step = window

        if not log:
            return ((price_mat/np.roll(price_mat, shift=shift, axis=1)) - 1)[:, shift::step], index[shift::step]
        return np.log((price_mat/np.roll(price_mat, shift=shift, axis=1)))[:, shift::step], index[shift::step]

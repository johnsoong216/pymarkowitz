import numpy as np
import pandas as pd
from .Exceptions import *


class ReturnGenerator:

    def __init__(self, price_data, assets=None):

        if isinstance(price_data, pd.DataFrame):
            self.price_mat = price_data.values.T
            self.assets = price_data.columns
            self.index = price_data.index

        elif isinstance(price_data, np.ndarray):

            self.price_mat = price_data
            self.assets = assets
            self.index = np.arange(0, len(price_data), 1)

    def calc_return(self, method, ret_format='df', **kwargs):

        price_mat = self.price_mat
        index = self.index

        if method == 'daily':
            ret_mat, ret_idx = ReturnGenerator.return_formula(price_mat, index, window=1, roll=True, **kwargs)
        elif method == 'rolling':
            ret_mat, ret_idx = ReturnGenerator.return_formula(price_mat, index, roll=True, **kwargs)
        elif method == 'collapse':
            ret_mat, ret_idx = ReturnGenerator.return_formula(price_mat, index, roll=False, **kwargs)
        # May add later
        # elif method == 'week':
        #     pass
        # elif method == 'month':
        #     pass
        # elif method == 'quarter':
        #     pass
        # elif method == 'annual':
        #     pass
        else:
            raise MethodException("""Invalid Method. Valid Inputs: daily, rolling, collapse""")

        return_df = pd.DataFrame(ret_mat.T, columns=self.assets, index=ret_idx)

        if ret_format == 'df':
            return return_df
        elif ret_format == 'raw':
            return self.assets, ret_idx, ret_mat
        else:
            raise FormatException("Invalid Format. Valid options are: df, raw")

    def calc_mean_return(self, method, time_scaling, ret_format='series', **kwargs):

        price_mat = self.price_mat
        index = self.index

        ret_mat, ret_idx = ReturnGenerator.return_formula(price_mat, index, **kwargs)

        if method == 'arithmetic':
            mean_ret = np.mean(ret_mat, axis=1) * time_scaling
        elif method == 'geometric':
            mean_ret = (price_mat[:, -1]/price_mat[:, 0]) ** (1/price_mat.shape[0]) - 1
        else:
            raise MethodException("""Method options are arithmetic/geometric""")

        if ret_format == 'series':
            return pd.Series(mean_ret, index=self.assets)
        elif ret_format == "raw":
            return mean_ret, self.assets
        else:
            raise FormatException("""Invalid Format. Valid options are: series, raw""")




    @staticmethod
    def return_formula(price_mat, index, roll=True, window=1, log=False):

        if roll:
            step = 1
            shift = window
        else:
            shift = window
            step = window

        if not log:
            return ((price_mat/np.roll(price_mat, shift=shift, axis=1)) - 1)[:, shift::step], index[shift::step]
        return np.log((price_mat/np.roll(price_mat, shift=shift, axis=1)))[:, shift::step], index[shift::step]

    # def result(self, ret_mat=None, index=None, return_format='df', **kwargs):
    #
    #     if ret_mat is None:
    #         ret_mat = self.return_mat
    #     if index is None:
    #         index = self.return_index
    #
    #     df = pd.DataFrame(ret_mat.T, columns=self.assets, index=index, **kwargs)
    #
    #     if return_format == 'df':
    #         return df
    #     elif return_format == 'dict':
    #         return df.unstack().to_dict()
    #     elif return_format == 'raw':
    #         return self.assets, ret_mat, index
    #     else:
    #         raise FormatException("Invalid Format. Valid options are: df, dict, raw")

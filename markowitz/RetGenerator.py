import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
from .Exceptions import *


class RetGenerator:

    def __init__(self, price_data, assets=None):

        if isinstance(price_data, pd.DataFrame):

            ### Detect Non-Numeric Column

            self.return_mat = price_data.pct_change().dropna(how='any').values.transpose()
            self.assets = price_data.columns

        elif isinstance(price_data, np.ndarray):

            self.return_mat = np.diff(price_data) / price_data[: ,1:]
            self.assets = assets

    def exp_smoothing(self, decay):

        dim = self.return_mat.shape
        decay_mat = decay ** np.linspace(tuple(np.arange(0 ,dim[1])), tuple(np.arange(0 ,dim[1])), dim[0])

        self.return_mat = np.multiply(self.return_mat, decay_mat)

    def return_mat(self, return_format='df'):

        df = pd.DataFrame(self.return_mat, columns=self.assets)

        if return_format == 'df':
            return df
        elif return_format == 'timeseries':
            return
        elif return_format == 'dist':
            return
        elif return_format == 'raw':
            return self.assets, self.return_mat
        else:
            raise FormatException("Invalid Format. Valid options are: df, timeseries, dict, raw")

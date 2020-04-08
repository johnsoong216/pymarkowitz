import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
from .Exceptions import *


class CovGenerator:

    def __init__(self, return_data, assets=None):

        if isinstance(return_data, pd.DataFrame):
            self.return_mat = return_data.values.transpose()
            self.assets = return_data.columns
        elif isinstance(return_data, np.ndarray):
            self.return_mat = return_data
            self.assets = assets

        self.cov_mat = None

    def calc_cov_mat(self, technique='sample', semi=False, method='default', unit_time=250, bm_return=0,
                     assume_zero=False, decay=0.95, **kwargs):

        return_mat = self.return_mat

        if semi:
            return_mat = self.semi_cov(return_mat, bm_return, assume_zero)

        if technique == "sample":
            self.cov_mat = self.sample_cov(return_mat, method, unit_time, decay)
        else:
            self.cov_mat = self.sk_technique(return_mat, unit_time, **kwargs)

    def sk_technique(self, return_mat, technique, unit_time, **kwargs):

        technique_dict = {"EmpiricalCovariance": sk.covariance.EmpiricalCovariance,
                          "EllipticEnvelope": sk.covariance.EllipticEnvelope,
                          "GraphicalLasso": sk.covariance.GraphicalLasso,
                          "GraphicalLassoCV": sk.covariance.GraphicalLassoCV,
                          "LedoitWolf": sk.covariance.LedoitWolf,
                          "MinCovDet": sk.covariance.MinCovDet,
                          "OAS": sk.covariance.OAS,
                          "ShrunkCovariance": sk.covariance.ShrunkCovariance}

        return technique_dict[technique](**kwargs).fit(return_mat.T).covariance_ * unit_time

    def sample_cov(self, return_mat, method, unit_time, decay):

        if method == 'default':
            cov_mat = self.default_cov(return_mat, unit_time)
        elif method == 'exp':
            cov_mat = self.exp_cov(return_mat, unit_time=unit_time, decay=decay)
        elif method == 'garch':
            cov_mat = self.garch_cov(return_mat, unit_time)
        else:
            raise MethodException("Invalid Method, Valid options are: default, exp, garch")
        return cov_mat

    def semi_cov(self, return_mat, bm_return, assume_zero):

        _return_mat_copy = return_mat.copy()

        def adj_return_vec(return_vec, bm_return):
            """
            Does not assume daily mean return of 0
            """
            return_vec[return_vec >= bm_return] = np.mean(return_vec[return_vec < bm_return])
            return return_vec

        if not assume_zero:
            _return_mat_copy = np.apply_along_axis(adj_return_vec, axis=1, arr=_return_mat_copy, bm_return=bm_return)
        else:
            _return_mat_copy = np.fmin(_return_mat_copy - bm_return, 0)

        return _return_mat_copy

    def default_cov(self, return_mat, unit_time):

        """
        Covariance Matrix without Exponential Decay
        """
        return np.cov(return_mat) * unit_time

    def exp_cov(self, return_mat, decay, unit_time):

        """
        Covariance Matrix with Exponential Decay
        """

        exp_weights = decay ** np.arange(0, return_mat.shape[1])

        return np.cov(return_mat, aweights=exp_weights) * unit_time

    def garch_cov(self, return_mat):
        pass

    def return_mat(self, corr=True, return_format='df', **kwargs):

        if corr:
            res_mat = self.cov_mat * np.dot(((np.diag(self.cov_mat)) ** -0.5).reshape(-1, 1),
                                       ((np.diag(self.cov_mat)) ** -0.5).reshape(1, -1))
        else:
            res_mat = self.cov_mat

        df = pd.DataFrame(res_mat, index=self.assets, columns=self.assets)

        if return_format == 'df':
            return df
        elif return_format == 'dict':
            return df.unstack().to_dict()
        elif return_format == 'heatmap':
            return sns.heatmap(df, **kwargs)
        elif return_format == 'dist':
            return sns.distplot(res_mat[np.tril_indices(res_mat.shape[0])], **kwargs)
        elif return_format == 'raw':
            return self.assets, res_mat
        else:
            raise FormatException("Invalid Format. Valid options are: df, dict, heatmap, dist, raw")
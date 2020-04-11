import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import warnings
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

    def calc_cov_mat(self, technique='sample', semi=False, method='default', unit_time=250,
                     inplace=True, builtin=False, weights=None, bm_return=0.001, assume_zero=False, **kwargs):

        return_mat = self.return_mat

        if semi:
            return_mat = self.semi_cov(return_mat, bm_return=bm_return, assume_zero=assume_zero)

        if technique == "sample":
            cov_mat = self.sample_cov(return_mat, method, unit_time, builtin=builtin, weights=weights, **kwargs)
        else:
            cov_mat = self.sk_technique(return_mat, technique, unit_time, **kwargs)

        if inplace:
            self.cov_mat = cov_mat
        else:
            return self.result(cov_mat, corr=False)

    def sk_technique(self, return_mat, technique, unit_time=250, **kwargs):

        technique_dict = {"EmpiricalCovariance": sk.covariance.EmpiricalCovariance,
                          "EllipticEnvelope": sk.covariance.EllipticEnvelope,
                          "GraphicalLasso": sk.covariance.GraphicalLasso,
                          "GraphicalLassoCV": sk.covariance.GraphicalLassoCV,
                          "LedoitWolf": sk.covariance.LedoitWolf,
                          "MinCovDet": sk.covariance.MinCovDet,
                          "OAS": sk.covariance.OAS,
                          "ShrunkCovariance": sk.covariance.ShrunkCovariance}
        try:
            return technique_dict[technique](**kwargs).fit(return_mat.T).covariance_ * unit_time
        except KeyError:
            raise MethodException("""Invalid Technique. Options are EmpiricalCovariance, 
                                  EllipticEnvelope, GraphicalLasso, GraphicalLassoCV, LedoitWolf, MinCovDet,
                                  OAS, ShrunkCovariance""")

    def sample_cov(self, return_mat, method, unit_time, weights=None, builtin=False, **kwargs):

        if method == 'default':
            weights = np.repeat(np.divide(1, return_mat.shape[1]), repeats=return_mat.shape[1])
        elif method == 'exp':
            weights = CovGenerator.exp_factor(return_mat, **kwargs)
        elif method == 'custom':
            if weights is None:
                warnings.warn("""Weight factor not defined. will use equal weight to calculate covariance.""")
                weights = np.repeat(1 / return_mat.shape[1], repeats=return_mat.shape[1])
            elif weights.shape[0] > return_mat.shape[1]:
                warnings.warn(f"""Weight factor vector is longer than return series shape. Will only use {return_mat.shape[1]} values""")
                weights = weights[:return_mat.shape[1]]
            elif weights.shape[0] < return_mat.shape[1]:
                warnings.warn(f"""Weight factor vector is shorter than return series shape. Will default remaining weights to 0""")
                weights = np.pad(weights, (0, return_mat.shape[1] - weights.shape[0]), 'constant')
        else:
            raise MethodException("""Invalid Method, Valid options are: default (equal weight), exp (exponential decay),
                                    custom (custom weight)""" )

        return CovGenerator.find_cov(return_mat, weights, builtin) * unit_time

    def semi_cov(self, return_mat, bm_return=0.0001, assume_zero=False):

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

    @staticmethod
    def find_cov(return_mat, weight_factor, builtin):

        """
        Covariance Matrix without Exponential Decay

        raw implementation as opposed to calling np.cov()
        """
        if builtin:
            return np.cov(return_mat, aweights=weight_factor)

        diff_mat = return_mat - np.mean(return_mat, axis=1, keepdims=True)
        weight_mat = CovGenerator.calc_weight_mat(return_mat, weight_factor)
        return np.dot(weight_mat * diff_mat, diff_mat.T)

    @staticmethod
    def calc_weight_mat(return_mat, weight_factor):
        """

        :param return_mat:
        :param weight_factor:
        :return:
        """
        weights = weight_factor/np.sum(weight_factor)
        weight_mat = np.repeat(weights.reshape(1, -1), repeats=return_mat.shape[0], axis=0)
        return weight_mat

    @staticmethod
    def exp_factor(return_mat, decay=0.94, span=30):

        """
        Covariance Matrix with Exponential Decay
        """

        decay_factor = decay ** np.arange(0, return_mat.shape[1]//span + 1)
        decay_factor = np.repeat(decay_factor, repeats=span)[:return_mat.shape[1]]

        return decay_factor

        # for col in range(num_assets):
        #     for row in range(col, num_assets):
        #         cov_mat[col, row] = np.sum(
        #             (return_mat[col, :] - return_mat[col, :].mean()) * \
        #             (return_mat[row, :] - return_mat[row, :].mean()) * exp_weights
        #         )
        #
        # return cov_mat + cov_mat.T - np.diag(cov_mat.diagonal())

    def result(self, cov_mat=None, corr=True, return_format='df', **kwargs):

        if cov_mat is None:
            cov_mat = self.cov_mat

        if corr:
            res_mat = cov_mat * np.dot(((np.diag(cov_mat)) ** -0.5).reshape(-1, 1),
                                       ((np.diag(cov_mat)) ** -0.5).reshape(1, -1))
        else:
            res_mat = cov_mat

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
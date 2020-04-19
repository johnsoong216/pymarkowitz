import numpy as np
import pandas as pd
import sklearn as sk
import itertools
import warnings
from .Exceptions import *


class MomentGen:

    def __init__(self, return_data, assets=None):

        if isinstance(return_data, pd.DataFrame):
            self.return_mat = return_data.values.transpose()
            self.assets = return_data.columns
        elif isinstance(return_data, np.ndarray):
            self.return_mat = return_data
            self.assets = assets

    def calc_cov_mat(self, technique='sample', semi=False, method='default', unit_time=252,
                     builtin=False, weights=None, bm_return=0.001, assume_zero=False, normalize=False, ret_format='df', **kwargs):

        return_mat = self.return_mat

        if semi:
            return_mat = self.semi_cov(return_mat, bm_return=bm_return, assume_zero=assume_zero)

        if technique == "sample":
            cov_mat = self.sample_cov(return_mat, method, unit_time, builtin=builtin, weights=weights, **kwargs)
        else:
            cov_mat = self.sk_technique(return_mat, technique, unit_time, **kwargs)

        if normalize:
            cov_mat = cov_mat * np.dot(((np.diag(cov_mat)) ** -0.5).reshape(-1, 1),
                                       ((np.diag(cov_mat)) ** -0.5).reshape(1, -1))
        if ret_format == 'df':
            return pd.DataFrame(cov_mat, index=self.assets, columns=self.assets)
        elif ret_format == 'raw':
            return self.assets, cov_mat
        else:
            raise FormatException("Invalid Format. Valid options are: df, raw")
            
    def calc_beta(self, beta_vec, technique="sample", semi=False, method='default', builtin=False, weights=None, ret_format='df', **kwargs):

        if isinstance(beta_vec, (pd.DataFrame, pd.Series)):
            beta_vec = beta_vec.values.T
        elif isinstance(beta_vec, np.ndarray):
            beta_vec = beta_vec.reshape(1, -1)

        if beta_vec.shape[1] != self.return_mat.shape[1]:
            raise DimException("""Dimension of benchmark (beta） data is not in the same length as the return data""")

        return_mat = np.concatenate([self.return_mat, beta_vec])

        if semi:
            return_mat = self.semi_cov(return_mat, bm_return=beta_vec.mean(), assume_zero=False)

        if technique == "sample":
            cov_mat = self.sample_cov(return_mat, method, unit_time=1, builtin=builtin, weights=weights, **kwargs)
        else:
            cov_mat = self.sk_technique(return_mat, technique, unit_time=1, **kwargs)


        beta_arr = cov_mat[-1, :-1]/(np.std(self.return_mat, ddof=1, axis=1)**2)

        if ret_format == 'df':
            return pd.DataFrame(beta_arr, index=self.assets, columns=['beta'])
        elif ret_format == 'raw':
            return self.assets, beta_arr
        else:
            raise FormatException("Invalid Format. Valid Options are: df, raw")


    def calc_coskew_mat(self, semi=False, method='default', weights=None, bm_return=0.001, assume_zero=False, normalize=True, ret_format='df', **kwargs):
        return self.calc_comoment_mat(moment=3, semi=semi, method=method, weights=weights, bm_return=bm_return, assume_zero=assume_zero, normalize=normalize, ret_format=ret_format, **kwargs)

    def calc_cokurt_mat(self, semi=False, method='default', weights=None, bm_return=0.001, assume_zero=False, normalize=True, ret_format='df', **kwargs):
        cokurt_mat =  self.calc_comoment_mat(moment=4, semi=semi, method=method, weights=weights, bm_return=bm_return, assume_zero=assume_zero, normalize=normalize, ret_format=ret_format, **kwargs)
        return cokurt_mat

    def calc_comoment_mat(self, moment, semi=False, method='default', weights=None, bm_return=0.001, assume_zero=False, normalize=True, ret_format='df', **kwargs):

        return_mat = self.return_mat

        if semi:
            return_mat = self.semi_cov(return_mat, bm_return=bm_return, assume_zero=assume_zero)

        weight_factor = self.construct_weight(method, return_mat, weights, **kwargs)

        weight_mat = MomentGen.calc_weight_mat(return_mat, weight_factor)
        comoment_mat = MomentGen.calc_moment_mat(moment, return_mat, weight_mat, normalize)

        if ret_format == 'df':
            return pd.DataFrame(comoment_mat, index=self.assets, columns=tuple(itertools.product(*[self.assets for i in range(moment - 1)])))
        elif ret_format == 'raw':
            return self.assets, comoment_mat
        else:
            raise FormatException("Invalid Format. Valid options are: df, raw")






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

        weights = self.construct_weight(method, return_mat, weights, **kwargs)
        return MomentGen.find_cov(return_mat, weights, builtin) * unit_time

    def construct_weight(self, method, return_mat, weights, **kwargs):
        if method == 'default':
            weights = np.repeat(np.divide(1, return_mat.shape[1]), repeats=return_mat.shape[1])
        elif method == 'exp':
            weights = MomentGen.exp_factor(return_mat, **kwargs)
        elif method == 'custom':
            if weights is None:
                warnings.warn("""Weight factor not defined. will use equal weight to calculate covariance.""")
                weights = np.repeat(1 / return_mat.shape[1], repeats=return_mat.shape[1])
            elif weights.shape[0] > return_mat.shape[1]:
                warnings.warn(
                    f"""Weight factor vector is longer than return series shape. Will only use {return_mat.shape[1]} values""")
                weights = weights[:return_mat.shape[1]]
            elif weights.shape[0] < return_mat.shape[1]:
                warnings.warn(
                    f"""Weight factor vector is shorter than return series shape. Will default remaining weights to 0""")
                weights = np.pad(weights, (0, return_mat.shape[1] - weights.shape[0]), 'constant')
        else:
            raise MethodException("""Invalid Method, Valid options are: default (equal weight), exp (exponential decay),
                                    custom (custom weight)""")
        return weights

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
        weight_mat = MomentGen.calc_weight_mat(return_mat, weight_factor)
        return np.dot(weight_mat * diff_mat * (return_mat.shape[1]/(return_mat.shape[1] - 1)), diff_mat.T)

    @staticmethod
    def calc_moment_mat(moment, return_mat, weight_mat, normalize):

        diff_mat = return_mat - np.mean(return_mat, axis=1, keepdims=True)
        num_obs = diff_mat.shape[1]
        num_assets = diff_mat.shape[0]

        temp_mat = diff_mat.T
        for iteration in range(moment - 2):
            temp_mat = np.kron(diff_mat.T, temp_mat)[::num_obs + 1, :]

        unbias_factor = (num_obs ** (moment - 1)) / (np.prod(num_obs - np.arange(1, moment, 1)))
        weighted_diff_mat = np.multiply(weight_mat, diff_mat) * unbias_factor
        moment_mat = np.dot(weighted_diff_mat, temp_mat)
        if normalize:
            std_arr = np.std(diff_mat, ddof=1, axis=1).reshape(-1, 1)
            temp_std_mat = std_arr.T
            for iteration in range(moment - 2):
                temp_std_mat = np.kron(std_arr.T, temp_std_mat)[::num_assets + 1, :]
            std_arr_mat = np.dot(std_arr, temp_std_mat)
            moment_mat = moment_mat / std_arr_mat
        return moment_mat

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

    # def result(self, cov_mat=None, corr=True, return_format='df', **kwargs):
    #
    #     if cov_mat is None:
    #         cov_mat = self.cov_mat
    #
    #     if corr:
    #         res_mat = cov_mat * np.dot(((np.diag(cov_mat)) ** -0.5).reshape(-1, 1),
    #                                    ((np.diag(cov_mat)) ** -0.5).reshape(1, -1))
    #     else:
    #         res_mat = cov_mat
    #
    #     df = pd.DataFrame(res_mat, index=self.assets, columns=self.assets)
    #
    #     if return_format == 'df':
    #         return df
    #     elif return_format == 'dict':
    #         return df.unstack().to_dict()
    #     elif return_format == 'heatmap':
    #         return sns.heatmap(df, **kwargs)
    #     elif return_format == 'dist':
    #         return sns.distplot(res_mat[np.tril_indices(res_mat.shape[0])], **kwargs)
    #     elif return_format == 'raw':
    #         return self.assets, res_mat
    #     else:
    #         raise FormatException("Invalid Format. Valid options are: df, dict, heatmap, dist, raw")
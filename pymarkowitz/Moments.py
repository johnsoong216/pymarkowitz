"""
MomentGenerator generates moment matrices used in mean-variance optimization problems.
Covariance Matrix (of Moment 2) and Higher Moment Matrices can all be generated using the class.

Higher Moment Implementation Reference
https://cran.r-project.org/web/packages/PerformanceAnalytics/vignettes/EstimationComoments.pdf
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import *

from .Exceptions import *


class MomentGenerator:

    def __init__(self, return_data, assets=None):
        """
        Initializes MomentGenerator
        :param return_data: pd.DataFrame/np.ndarray, return data with return values in percentage terms
        :param assets: List[str] optional, list of assets/tickers corresponding to the np.ndarray
        """

        if isinstance(return_data, pd.DataFrame):
            self.return_mat = return_data.values.transpose()
            self.assets = return_data.columns
        elif isinstance(return_data, np.ndarray):
            self.return_mat = return_data
            self.assets = assets
        else:
            raise FormatException("Invalid Format. Return Data options: pd.DataFrame, np.ndarray")

    def calc_cov_mat(self, technique='sample', semi=False, method='default', time_scaling=252,
                     builtin=False, weights=None, bm_return=0.00025, assume_zero=False, normalize=False, ret_format='df', **kwargs):
        """
        Calculates covariance matrix given the return data input.

        :param technique: str, default='sample'
                additional_options: ["EmpiricalCovariance", "EllipticEnvelope", "GraphicalLasso", "GraphicalLassoCV",
                                    "LedoitWolf", "MinCovDet", "OAS", "ShrunkCovariance"]
                Specifies the calculation technique for the covariance matrix
        :param semi: bool, default=False
                If True, returns a semivariance matrix that emphasizes on downside portfolio variance
        :param method: str, default='default'
                additional_options: ["exponential", "custom"]
                Default implies all returns are weighted equally,
                Exponential requires the specification of decay factors and timespan of decay
                Custom requires a customized weight array
        :param time_scaling: int, default=252
                Default annualizes the covariance matrix (assuming daily return is the input)
        :param builtin: bool, default=False,
                If True then calls np.cov() to calculate, otherwise use matrix calculation method written in the class.
                Note that calling builtin function versus using calculation methods yield identical result
        :param weights: np.ndarray, default=None
                If choose parameter method='custom', then input the customized weight array
        :param bm_return: float, default=0.00025,
                Additional parameter for calculating semivariance matrix.
                Ignores all individual asset returns above the bm_return when calculating covariance
        :param assume_zero: bool, default=False
                Additional parameter for calculating semivariance matrix.
                Long term daily mean return for an individual asset is sometiimes assumed to be 0.
                However, this may not hold true if sample size is not sufficiently large.
        :param normalize: bool, default=False
                To normalize the covariance matrix. In the specific case for covariance matrix, a normalized covariance
                matrix is a correlation matrix.
        :param ret_format: str, default='df'
                Additional Options: ["raw"]
                Returns a covariance matrix in the form of a pd.DataFrame, or a tuple of asset names and np.ndarray
        :param kwargs: decay, span (Check MomentGenerator.exp_factor for usage)
        :return: Covariance Matrix
        """

        return_mat = self.return_mat

        if semi:
            return_mat = self.semi_cov(return_mat, bm_return=bm_return, assume_zero=assume_zero)

        if technique == "sample":
            cov_mat = self.sample_cov(return_mat, method, time_scaling, builtin=builtin, weights=weights, **kwargs)
        else:
            cov_mat = self.sk_technique(return_mat, technique, time_scaling, **kwargs)

        if normalize:
            cov_mat = cov_mat * np.dot(((np.diag(cov_mat)) ** -0.5).reshape(-1, 1),
                                       ((np.diag(cov_mat)) ** -0.5).reshape(1, -1))
        if ret_format == 'df':
            return pd.DataFrame(cov_mat, index=self.assets, columns=self.assets)
        elif ret_format == 'raw':
            return self.assets, cov_mat
        else:
            raise FormatException("Invalid Format. Valid options are: df, raw")
            
    def calc_beta(self, beta_vec, technique="sample", semi=False, method='default', builtin=False, weights=None, ret_format='series', **kwargs):
        """
        Calculates beta of each asset given benchmark return.

        Beta = Covariance(Asset, Benchmark)/Variance(Benchmark)

        The parameters' usage are identical to that of calling method calc_cov_mat as the method creates a
        covariance matrix combining the benchmark return and the individual asset return.

        In this case, setting semi=True gives the downside beta.

        :param beta_vec: pd.DataFrame/pd.Series/np.ndarray, benchmark return vector
        :param ret_format: str, default='series'
                Additional option = ["raw"]
                Returns a covariance matrix in the form of a pd.Series, or a tuple of asset names and np.ndarray
        :return: beta values for each asset
        """

        if isinstance(beta_vec, (pd.DataFrame, pd.Series)):
            beta_vec = beta_vec.values.T.reshape(1, -1)
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
            cov_mat = self.sk_technique(return_mat, technique, time_scaling=1, **kwargs)

        beta_arr = cov_mat[-1, :-1]/cov_mat[-1, -1]

        if ret_format == 'series':
            return pd.Series(beta_arr, index=self.assets)
        elif ret_format == 'raw':
            return self.assets, beta_arr
        else:
            raise FormatException("Invalid Format. Valid Options are: series, raw")


    def calc_coskew_mat(self, semi=False, method='default', weights=None, bm_return=0.001, assume_zero=False, normalize=True, ret_format='df', **kwargs):

        """
        Calculates the coskewness matrix given the return data.
        Dimension (N, N^2) N: num_assets

        The parameters' usage are identical to that of calling method calc_cov_mat.
        Parameter normalize default=True as coskewness matrix is typically used in its normalized form
        """
        return self.calc_comoment_mat(moment=3, semi=semi, method=method, weights=weights, bm_return=bm_return, assume_zero=assume_zero, normalize=normalize, ret_format=ret_format, **kwargs)

    def calc_cokurt_mat(self, semi=False, method='default', weights=None, bm_return=0.001, assume_zero=False, normalize=True, ret_format='df', **kwargs):

        """
        Calculates the cokurtosis matrix given the return data.
        Dimension (N, N^3) N: num_assets

        The parameters' usage are identical to that of calling method calc_cov_mat.
        Parameter normalize default=True as cokurtosis matrix is typically used in its normalized form
        """
        data_size = len(self.return_mat[0])
        bias_factor = (data_size - 3) * (data_size - 2)/ data_size/(data_size - 1)  
        return self.calc_comoment_mat(moment=4, semi=semi, method=method, weights=weights, bm_return=bm_return, assume_zero=assume_zero, normalize=normalize, ret_format=ret_format, **kwargs) * bias_factor

    def calc_comoment_mat(self, moment, semi=False, method='default', weights=None, bm_return=0.001, assume_zero=False, normalize=True, ret_format='df', **kwargs):

        """
        Calculates higher moment matrices matrix given the return data.
        Dimension (N, N^(moment-1)) N: num_assets

        :param moment: int, moment number

        *Moments higher than 4 (cokurtosis) are not commonly used in portfolio constructions.
        *Calling this method to calculate covariance matrix by setting moment=2 will yield identical result

        The parameters' usage are identical to that of calling method calc_cov_mat.
        """
        return_mat = self.return_mat

        if semi:
            return_mat = self.semi_cov(return_mat, bm_return=bm_return, assume_zero=assume_zero)

        weight_factor = self.construct_weight(method, return_mat, weights, **kwargs)

        weight_mat = MomentGenerator.calc_weight_mat(return_mat, weight_factor)
        comoment_mat = MomentGenerator.calc_moment_mat(moment, return_mat, weight_mat, normalize)

        if ret_format == 'df':
            return pd.DataFrame(comoment_mat, index=self.assets, columns=tuple(itertools.product(*[self.assets for i in range(moment - 1)])))
        elif ret_format == 'raw':
            return self.assets, comoment_mat
        else:
            raise FormatException("Invalid Format. Valid options are: df, raw")

    def sk_technique(self, return_mat, technique, time_scaling=252, **kwargs):

        """
        Using sklearn.covariance methods to construct covariance matrix
        :param return_mat: np.ndarray, return matrix
        :param technique: str, options to select sklearn.covariance methods
        :param time_scaling: int, default=252, annualize covariance matrix (assuming daily input)
        :param kwargs: additional arguments used in sklearn methods
        :return: np.ndarray, covariance matrix in its raw form
        """
        technique_dict = {"EmpiricalCovariance": EmpiricalCovariance,
                          "EllipticEnvelope": EllipticEnvelope,
                          "GraphicalLasso": GraphicalLasso,
                          "GraphicalLassoCV": GraphicalLassoCV,
                          "LedoitWolf": LedoitWolf,
                          "MinCovDet": MinCovDet,
                          "OAS": OAS,
                          "ShrunkCovariance": ShrunkCovariance}
        try:
            return technique_dict[technique](**kwargs).fit(return_mat.T).covariance_ * time_scaling
        except KeyError:
            raise MethodException("""Invalid Technique. Options are EmpiricalCovariance, 
                                  EllipticEnvelope, GraphicalLasso, GraphicalLassoCV, LedoitWolf, MinCovDet,
                                  OAS, ShrunkCovariance""")

    def sample_cov(self, return_mat, method, unit_time, weights=None, builtin=False, **kwargs):

        """
        Calculates the sample covariance
        """

        weights = self.construct_weight(method, return_mat, weights, **kwargs)
        return MomentGenerator.find_cov(return_mat, weights, builtin) * unit_time

    def construct_weight(self, method, return_mat, weights, **kwargs):

        """
        Constructing return weight based on method
        """

        if method == 'default':
            weights = np.repeat(np.divide(1, return_mat.shape[1]), repeats=return_mat.shape[1])
        elif method == 'exp':
            weights = MomentGenerator.exp_factor(return_mat, **kwargs)
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
        # print(weights)
        return weights

    def semi_cov(self, return_mat, bm_return=0.0001, assume_zero=False):

        """
        Calculates semivariance matrix given bm_return
        """

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
        Calculates variance based on weight array and return matrix
        """

        if builtin:
            return np.cov(return_mat, aweights=weight_factor)

        diff_mat = return_mat - np.mean(return_mat, axis=1, keepdims=True)
        weight_mat = MomentGenerator.calc_weight_mat(return_mat, weight_factor)
        return np.dot(weight_mat * diff_mat * (return_mat.shape[1]/(return_mat.shape[1] - 1)), diff_mat.T)

    @staticmethod
    def calc_moment_mat(moment, return_mat, weight_mat, normalize):

        """
        Calculates moment matrix
        """

        # Difference
        diff_mat = return_mat - np.mean(return_mat, axis=1, keepdims=True)
        num_obs = diff_mat.shape[1]
        num_assets = diff_mat.shape[0]

        # Kronecker Product
        temp_mat = diff_mat.T
        for iteration in range(moment - 2):
            temp_mat = np.kron(diff_mat.T, temp_mat)[::num_obs + 1, :]

        # DDOF calculation (May cause overflow if moment is too high)
        unbias_factor = np.prod(np.repeat(num_obs, moment-1)/(num_obs - np.arange(1, moment, 1))) #(num_obs ** (moment - 1)) / (np.prod(num_obs - np.arange(1, moment, 1)))
        weighted_diff_mat = np.multiply(weight_mat, diff_mat) * unbias_factor
        moment_mat = np.dot(weighted_diff_mat, temp_mat)

        # Normalizing each value in the matrix with standard deviations
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
        Construct a weight matrix
        """
        weights = weight_factor/np.sum(weight_factor)
        weight_mat = np.repeat(weights.reshape(1, -1), repeats=return_mat.shape[0], axis=0)
        return weight_mat

    @staticmethod
    def exp_factor(return_mat, decay=0.94, span=30):
        """
        Constructs weight factor based on decay and span

        A decay of 0.94 and a span of 2 will construct weight factor as based on proximity to benchmark date
        as [0.94, 0.94, 0.94^2, 0.94^2, ...]
        """
        decay_factor = decay ** np.arange(0, return_mat.shape[1]//span + 1)
        decay_factor = np.repeat(decay_factor, repeats=span)[:return_mat.shape[1]]
        return decay_factor[::-1]

import math
import warnings

import numpy as np
import pandas as pd
import cvxpy as cp
import pandas_datareader as data

from .Exceptions import *
### This class hosts all the metrics that can be used for both objectives/constraints
# https://investresolve.com/file/pdf/Portfolio-Optimization-Whitepaper.pdf

class MetricGen:



    def __init__(self, weight_param, ret_vec, moment_mat, moment, assets):
        self.weight_param = weight_param
        self.ret_vec = ret_vec
        self.moment_mat = moment_mat
        self.moment = moment
        self.assets = assets

    def expected_return(self):
        return self.weight_param * self.ret_vec

    def correlation(self):
        # Assume Covariance Matrix is passed in
        corr_mat = self.moment_mat * np.dot(((np.diag(self.moment_mat)) ** -0.5).reshape(-1, 1),
                                            ((np.diag(self.moment_mat)) ** -0.5).reshape(1, -1))
        return cp.quad_form(self.weight_param, corr_mat)

    def diversification(self):
        if self.moment != 2:
            raise DimException("Did not pass in a covariance matrix")
        std_arr = np.diag(self.moment_mat) ** 0.5
        return self.weight_param * std_arr/cp.sqrt(cp.quad_form(self.weight_param, self.moment_mat))

    def variance(self):
        if self.moment != 2:
            raise DimException("Did not pass in a covariance matrix")
        return cp.quad_form(self.weight_param, self.moment_mat)

    def volatility(self):
        return cp.sqrt(self.variance())

    def higher_moment(self, moment):
        temp = self.weight_param
        for iteration in range(moment - 2):
            temp = cp.kron(self.weight_param, temp)

        return self.weight_param @ self.moment_mat @ temp

    def risk_parity(self):
        return 0.5 * cp.quad_form(self.weight_param, self.moment_mat) - cp.sum(cp.log(self.weight_param))/len(self.assets)


    def market_cap_data(self):
        tickers = self.assets
        return data.get_quote_yahoo(tickers)['marketCap'].values

    def construct_bound(self, bound):
        pass

    #
    def inverse_volatility(self):
        pass
    #
    def inverse_variance(self):
        pass

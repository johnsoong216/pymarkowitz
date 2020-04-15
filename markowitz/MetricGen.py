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

    # Involves Risk Matrix Only

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


    # Involves Return
    def expected_return(self):
        return self.weight_param * self.ret_vec

    # Sortino if passed in a semivariance matrix
    def sharpe(self, risk_free):
        return (self.expected_return() - risk_free)/self.variance()

    def beta(self, individual_beta):
        return self.weight_param.T * individual_beta

    def treynor(self, risk_free, individual_beta):
        return (self.expected_return() - risk_free)/self.beta(individual_beta)

    def jenson_alpha(self, risk_free, market_return, individual_beta):
        return self.expected_return() - risk_free - self.beta(individual_beta) * (market_return - risk_free)

    # Tracking Error/ Calmar ratio /Omega not feasible (Can be included in backtesting)



    # Does not Involve the Weight Parameter

    def inverse_volatility(self):
        std_arr = np.diag(self.moment_mat) ** 0.5
        return (1/std_arr)/np.sum(1/std_arr)
    #
    def inverse_variance(self):
        var_arr = np.diag(self.moment_mat)
        return (1/var_arr)/np.sum(1/var_arr)

    # Long only
    def equal_weight(self, leverage):
        return np.repeat(leverage/len(self.assets), len(self.assets))

    # Long only
    def market_cap_weight(self, leverage):
        market_cap_info = self.market_cap_data()
        return market_cap_info/np.sum(market_cap_info) * leverage



    # Helper Function
    def market_cap_data(self):
        tickers = self.assets
        return data.get_quote_yahoo(tickers)['marketCap'].values




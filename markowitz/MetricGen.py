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



    def __init__(self, ret_vec, moment_mat, moment, assets, beta_vec):

        self.ret_vec = ret_vec
        self.moment_mat = moment_mat
        self.moment = int(moment)
        self.assets = assets
        self.beta_vec = beta_vec

        self.method_dict = {"leverage": self.leverage,
                            "num_assets": self.num_assets,
                            "concentration": self.concentration,
                            "correlation": self.correlation,
                            "diversification": self.diversification,
                            "variance": self.variance,
                            "volatility": self.volatility,
                            "skew": self.higher_moment,
                            "kurt": self.higher_moment,
                            "moment": self.higher_moment,
                            "risk_parity": self.risk_parity,
                            "expected_return": self.expected_return,
                            "beta": self.beta,
                            "treynor": self.treynor,
                            "jenson_alpha": self.jenson_alpha}

    # Weight Related
    def leverage(self, w):
        return np.sum(np.sqrt(np.square(w)))

    def num_assets(self, w):
        return len(w[np.round(w, 3) != 0])

    def concentration(self, w, top_holdings):
        return -np.sum(np.partition(-np.sqrt(np.square(w)), top_holdings)[:top_holdings])/np.sum(np.sqrt(np.square(w)))
            # return [{"type": "ineq", "fun": lambda w: np.sum(
            #     np.partition(-np.sqrt(np.square(w)), top_holdings)[:top_holdings]) / np.sum(
            #     np.sqrt(np.square(w))) + top_concentration}]

    # Involves Risk Only
    def correlation(self, w):
        # Assume Covariance Matrix is passed in
        corr_mat = self.moment_mat * np.dot(((np.diag(self.moment_mat)) ** -0.5).reshape(-1, 1),
                                            ((np.diag(self.moment_mat)) ** -0.5).reshape(1, -1))
        return w @ corr_mat @ w.T
        # return cp.quad_form(self.weight_param, corr_mat)

    def diversification(self, w):
        if self.moment != 2:
            raise DimException("Did not pass in a covariance matrix")
        std_arr = np.diag(self.moment_mat) ** 0.5
        return (w @ std_arr)/np.sqrt(w @ self.moment_mat @ w.T)

    def variance(self, w):
        if self.moment != 2:
            raise DimException("Did not pass in a covariance matrix")

        return w @ self.moment_mat @ w.T

    def volatility(self, w):
        return np.sqrt(self.variance(w))

    def higher_moment(self, w):
        temp = w
        for iteration in range(self.moment - 2):
            temp = np.kron(w, temp)
            # print(temp.shape)
            # print(self.moment_mat.shape)
            # print(w.shape)

        return w @ self.moment_mat @ temp

    def risk_parity(self, w):

        return 0.5 * w @ self.moment_mat @ w.T - np.sum(np.log(w))/len(self.assets)
        # return 0.5 * cp.quad_form(self.weight_param, self.moment_mat) - cp.sum(cp.log(self.weight_param))/len(self.assets)

    # Involves Return
    def expected_return(self, w):
        return w @ self.ret_vec

    # Sortino if passed in a semivariance matrix
    def sharpe(self, w, risk_free):
        return (self.expected_return(w) - risk_free)/self.volatility(w)

    def beta(self, w, individual_beta):
        return w @ individual_beta

    def treynor(self, w, risk_free, individual_beta):
        return (self.expected_return(w) - risk_free)/self.beta(w, individual_beta)

    def jenson_alpha(self, w, risk_free, market_return, individual_beta):
        return self.expected_return(w) - risk_free - self.beta(w, individual_beta) * (market_return - risk_free)

    # Tracking Error/ Calmar ratio /Omega not feasible (Can be included in backtesting)

    # Does not Involve Optimization

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




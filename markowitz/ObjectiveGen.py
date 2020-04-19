import numpy as np
import pandas as pd
import cvxpy as cp
import math
import warnings
from .Exceptions import *
from .MetricGen import MetricGen


class ObjectiveGen(MetricGen):

    def __init__(self, weight_param, ret_vec, moment_mat, moment, assets):

        super().__init__(weight_param, ret_vec, moment_mat, moment, assets)
        self.method_dict = {"efficient_frontier": self.efficient_frontier,
                            "equal_risk_parity": self.equal_risk_parity,
                            "min_correlation": self.min_correlation,
                            "min_volatility": self.min_volatility,
                            "min_variance": self.min_variance,
                            "min_skew": self.min_skew,
                            "min_kurt": self.min_kurt,
                            "min_moment": self.min_moment,
                            "max_diversification": self.max_diversification,
                            "max_sharpe": self.max_sharpe,
                            "min_beta": self.min_beta,
                            "max_treynor": self.max_treynor,
                            "max_jenson_alpha": self.max_jenson_alpha}

    def create_objective(self, objective_type):
        return self.method_dict[objective_type]

    # Classic Equation
    def efficient_frontier(self, w, aversion=1):
        return -(self.expected_return(w) - aversion * self.variance(w))

    # Risk Related
    def equal_risk_parity(self, w):
        return self.risk_parity(w)

    def min_correlation(self, w):
        return self.correlation(w)

    def min_volatility(self, w):
        return self.volatility(w)

    def min_variance(self, w):
        return self.variance(w)

    def min_skew(self, w):
        return self.min_moment(w)

    def min_kurt(self, w):
        return self.min_moment(w)

    def min_moment(self, w):
        return self.higher_moment(w)

    def max_diversification(self, w):
        return -self.diversification(w)

    # Metrics related
    def max_sharpe(self, w, risk_free):
        return -self.sharpe(w, risk_free)

    # Make beta close to zero
    def min_beta(self, w, individual_beta):
        return -np.abs(self.beta(w, individual_beta))

    def max_treynor(self, w, risk_free, individual_beta):
        return -self.treynor(w, risk_free, individual_beta)

    def max_jenson_alpha(self, w, risk_free, market_return, individual_beta):
        return -self.jenson_alpha(w, risk_free, market_return, individual_beta)



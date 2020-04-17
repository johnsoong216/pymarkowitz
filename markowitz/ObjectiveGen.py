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

    def create_objective(self, objective_type, **kwargs):
        return self.method_dict[objective_type](**kwargs)

    # Classic Equation
    def efficient_frontier(self, aversion=1):
        return cp.Maximize(self.expected_return() - aversion * self.variance())

    # Risk Related
    def equal_risk_parity(self):
        return cp.Minimize(self.risk_parity())

    def min_correlation(self):
        return cp.Minimize(self.correlation())

    def min_volatility(self):
        return cp.Minimize(self.volatility())

    def min_variance(self):
        return cp.Minimize(self.variance())

    def min_skew(self):
        return cp.Minimize(self.higher_moment(3))

    def min_kurt(self):
        return cp.Minimize(self.higher_moment(4))

    def min_moment(self):
        return cp.Minimize(self.higher_moment(self.moment))

    def max_diversification(self):
        return cp.Maximize(self.diversification())

    # Metrics related
    def max_sharpe(self, risk_free):
        return cp.Maximize(self.sharpe(risk_free))

    # Make beta close to zero
    def min_beta(self, individual_beta):
        return cp.Minimize(cp.abs(self.beta(individual_beta)))

    def max_treynor(self, risk_free, individual_beta):
        return cp.Maximize(self.treynor(risk_free, individual_beta))

    def max_jenson_alpha(self, risk_free, market_return, individual_beta):
        return cp.Maximize(self.jenson_alpha(risk_free, market_return, individual_beta))



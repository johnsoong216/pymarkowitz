import math
import warnings

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import *
import pandas_datareader as data

from .Exceptions import *
from .MetricGen import *

### https://www.portfolioprobe.com/features/constraints/


class ConstraintGen(MetricGen):


    def __init__(self, ret_vec, moment_mat, moment, assets, beta_vec):

        super().__init__(ret_vec, moment_mat, moment, assets, beta_vec)
        self.method_dict = {"weight": self.weight,
                            "num_assets": self.num_assets_const,
                            "concentration": self.concentration_const,
                            "market_neutral": self.market_neutral_const,
                            "expected_return": self.expected_return_const,
                            "sharpe": self.sharpe_const,
                            "beta": self.beta_const,
                            "treynor": self.treynor_const,
                            "jenson_alpha": self.jenson_alpha_const,
                            "volatility": self.volatility_const,
                            "variance": self.variance_const,
                            "skew": self.skew_const,
                            "kurt": self.kurt_const,
                            "moment": self.moment_const}

    def create_constraint(self, constraint_type, **kwargs):
        return self.method_dict[constraint_type](**kwargs)

    # Weight Only
    def weight(self, weight_bound, leverage): # Checked
        individual_bound = (0,1)
        if isinstance(weight_bound, (list, tuple)):
            if isinstance(weight_bound[0], (list, tuple)):
                if all([len(ind_weights) == 2 for ind_weights in weight_bound]) and len(weight_bound) == self.ret_vec.shape[0]:
                    weight_bound = np.array(weight_bound)
                else:
                    raise DimException("""If specifying weight for each individual asset, must be passed in pairs and 
                                            its length must equal the number of assets""")
            if isinstance(weight_bound[0], (float, int)):
                # constraints += [self.weight_param >= weight_bound[0]]
                # constraints += [self.weight_param <= weight_bound[1]]
                individual_bound = list(zip(np.repeat(weight_bound[0], self.ret_vec.shape[0]), np.repeat(weight_bound[1], self.ret_vec.shape[0])))
            else:
                raise FormatException("""Please pass in weight boundaries in an accepted format. List/Tuple/Np.ndarray""")
        if isinstance(weight_bound, np.ndarray):
            if weight_bound.ndim == 1:
                individual_bound = list(zip(np.repeat(weight_bound[0], self.ret_vec.shape[0]), np.repeat(weight_bound[1], self.ret_vec.shape[0])))

            elif weight_bound.ndim == 2:
                if weight_bound.shape != (self.ret_vec.shape[0], 2):
                    raise DimException("Dimension of Weights does not match number of assets")
                individual_bound = list(zip(weight_bound[:, 0], weight_bound[:, 1]))

            else:
                raise DimException("Dimension of Weight Bound Array must be 1/2")

        # total_bound = [{'type': 'eq', 'fun': lambda w:  np.sum(w) - total_weight}]
        total_leverage = [{'type': 'eq', 'fun': lambda w: -self.leverage(w) + leverage}]
        return individual_bound, total_leverage

    # def leverage(self, leverage): # Checked
    #     return [{'type': 'ineq', 'fun': lambda w: -np.sum(np.sqrt(np.square(w))) + leverage}]

    def num_assets_const(self, num_assets): # Checked
        if self.ret_vec.shape[0] <= num_assets:
            warnings.warn("""The number of assets to hold exceeds the number of assets available, 
            default to a 1 asset only scenario""")
            num_assets = self.ret_vec.shape[0] - 1
        non_holdings = self.ret_vec.shape[0] - num_assets
        # return [{'type': 'eq', 'fun': lambda w: self.num_assets(w) - num_assets}]
        return [{'type': 'eq', 'fun': lambda w: np.sum(np.partition(np.sqrt(np.square(w)), non_holdings)[:non_holdings])}]
        # return [cp.sum_smallest(cp.abs(self.weight_param), self.ret_vec.shape[0] - num_assets) <= 0.0001]

    def concentration_const(self, top_holdings, top_concentration): # Checked
        if self.ret_vec.shape[0] <= top_holdings:
            warnings.warn("""Number of Top Holdings Exceeds Total Available Assets. 
            Will default top_holdings to be number of holdings available""")
            top_holdings = self.ret_vec.shape[0]
        # return [{"type": "ineq", "fun": lambda w: -self.concentration(w, top_holdings) + top_concentration}]
        # return [cp.sum_largest(cp.norm(self.weight_param, 1), top_holdings) <= top_concentration]
        return [{"type": "ineq", "fun": lambda w: np.sum(
            np.partition(-np.sqrt(np.square(w)), top_holdings)[:top_holdings]) / np.sum(
            np.sqrt(np.square(w))) + top_concentration}]


    ### Market Data Needed/Calculation Needed
    def market_neutral_const(self, bound):

        market_cap_weight = self.market_cap_data()
        return [{"type": "ineq", "fun": lambda w: market_cap_weight @ self.moment_mat @ w - bound[0]},
                {"type": "ineq", "fun": lambda w: -(market_cap_weight @ self.moment_mat @ w - bound[1])}]
        # return [market_cap_weight @ self.moment_mat @ self.weight_param >= bound[0],
        #         market_cap_weight @ self.moment_mat @ self.weight_param <= bound[1]]

    # Return related
    def expected_return_const(self, bound): # Checked
        bound = self.construct_bound(bound, True, 10)
        # print(lambda w: self.expected_return(w) - bound[0])
        return [{"type": "ineq", "fun": lambda w: self.expected_return(w) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.expected_return(w) + bound[1]}]
        # return [{"type": "ineq", "fun": lambda w: np.power(1 + self.expected_return(w), time_scaling) - bound[0]},
        #         {"type": "ineq", "fun": lambda w: -np.power(1 + self.expected_return(w), time_scaling) + bound[1]}]

        # return [cp.power(1 + self.expected_return(), time_scaling) - 1 >= bound[0], cp.power(1 + self.expected_return(), time_scaling) - 1 <= bound[1]]

    def sharpe_const(self, risk_free, bound):
        bound = self.construct_bound(bound, True, 10)

        return [{"type": "ineq", "fun": lambda w: self.sharpe(w, risk_free) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.sharpe(w, risk_free) + bound[1]}]
        # return [self.sharpe(risk_free) >= bound[0], self.sharpe(risk_free) <= bound[1]]

    def beta_const(self, bound):
        bound = self.construct_bound(bound, False, 1)
        # return [self.beta(self.beta_vec) >= bound[0], self.beta(self.beta_vec) <= bound[1]]
        return [{"type": "ineq", "fun": lambda w: self.beta(w, self.beta_vec) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.beta(w, self.beta_vec) + bound[1]}]

    def treynor_const(self, bound, risk_free):
        bound = self.construct_bound(bound, True, 10)
        return [{"type": "ineq", "fun": lambda w: self.treynor(w, risk_free, self.beta_vec) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.treynor(w, risk_free, self.beta_vec)  + bound[1]}]
        # return [self.treynor(risk_free, self.beta_vec) >= bound[0], self.treynor(risk_free, self.beta_vec) <= bound[1]]

    def jenson_alpha_const(self, bound, risk_free, market_return):
        bound = self.construct_bound(bound, True, 10)
        return [{"type": "ineq", "fun": lambda w: self.jenson_alpha(w, risk_free, market_return, self.beta_vec) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.jenson_alpha(w, risk_free, market_return, self.beta_vec) + bound[1]}]
        # return [self.jenson_alpha(wrisk_free, market_return, self.beta_vec) >= bound[0], self.jenson_alpha(risk_free, market_return, self.beta_vec) <= bound[1]]

    # Risk related constraints
    def volatility_const(self, bound):
        bound = self.construct_bound(bound, False, 0)
        return [{"type": "ineq", "fun": lambda w: self.volatility(w)  + bound[0]},
                {"type": "ineq", "fun": lambda w: -self.volatility(w) + bound[1]}]
        # return [self.volatility() >= bound[0], self.volatility <= bound[1]]

    def variance_const(self, bound):
        if self.moment != 2:
            raise DimException("Did not pass in a correlation/covariance matrix")
        bound = self.construct_bound(bound, False, 0)
        return [{"type": "ineq", "fun": lambda w: self.variance(w) + bound[0]},
                {"type": "ineq", "fun": lambda w: -self.variance(w) + bound[1]}]

    def skew_const(self, bound):
        if self.moment != 3:
            raise DimException("Did not pass in a coskewness matrix")
        bound = self.construct_bound(bound, False, 0)
        return [{"type": "ineq", "fun": lambda w: self.higher_moment(w) + bound[0]},
                {"type": "ineq", "fun": lambda w: -self.higher_moment(w) + bound[1]}]

    def kurt_const(self, bound):
        if self.moment != 4:
            raise DimException("Did not pass in a cokurtosis matrix")

        bound = self.construct_bound(bound, False, 0)
        return [{"type": "ineq", "fun": lambda w: self.higher_moment(w) + bound[0]},
                {"type": "ineq", "fun": lambda w: -self.higher_moment(w) + bound[1]}]

    def moment_const(self, bound):
        bound = self.construct_bound(bound, False, 0)
        return [{"type": "ineq", "fun": lambda w: self.higher_moment(w) + bound[0]},
                {"type": "ineq", "fun": lambda w: -self.higher_moment(w) + bound[1]}]

    def construct_bound(self, bound, minimum, opposite_value):
        if isinstance(bound, (int, float)):
            warnings.warn(f"""Only one bound is given, will set the {'maximum' if minimum else 'minimum'} value to be {opposite_value}""")
            if minimum:
                bound = (bound, opposite_value)
            else:
                bound = (opposite_value, bound)
        return bound




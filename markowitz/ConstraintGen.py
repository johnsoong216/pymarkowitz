import math
import warnings

import numpy as np
import pandas as pd
import cvxpy as cp
import pandas_datareader as data

from .Exceptions import *
from .MetricGen import *

### https://www.portfolioprobe.com/features/constraints/


class ConstraintGen(MetricGen):


    def __init__(self, weight_param, ret_vec, moment_mat, moment, assets):

        super().__init__(weight_param, ret_vec, moment_mat, moment, assets)
        self.method_dict = {"weight": self.weight,
                            "leverage": self.leverage,
                            "num_assets": self.num_assets,
                            "concentration": self.concentration,
                            "market_neutral": self.market_neutral,
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
    def weight(self, weight_bound, total_weight):

        constraints = []
        if isinstance(weight_bound, (list, tuple)):
            if isinstance(weight_bound[0], (list, tuple)):
                if all([len(ind_weights) == 2 for ind_weights in weight_bound]) and len(weight_bound) == self.ret_vec.shape[0]:
                    weight_bound = np.array(weight_bound)
                else:
                    raise DimException("""If specifying weight for each individual asset, must be passed in pairs and 
                                            its length must equal the number of assets""")
            elif isinstance(weight_bound[0], (float, int)):
                constraints += [self.weight_param >= weight_bound[0]]
                constraints += [self.weight_param <= weight_bound[1]]
            else:
                raise FormatException("""Please pass in weight boundaries in an accepted format. List/Tuple/Np.ndarray""")

        if isinstance(weight_bound, np.ndarray):
            if weight_bound.ndim == 1:
                constraints += [self.weight_param >= weight_bound[0]]
                constraints += [self.weight_param <= weight_bound[1]]
            elif weight_bound.ndim == 2:
                if weight_bound.shape != (self.ret_vec.shape[0], 2):
                    raise DimException("Dimension of Weights does not match number of assets")
                constraints += [self.weight_param >= weight_bound[:, 0]]
                constraints += [self.weight_param <= weight_bound[:, 1]]
            else:
                raise DimException("Dimension of Weight Bound Array must be 1/2")
        else:
            raise FormatException("Weight bound needs to be in list/tuple/np.ndarray format")

        constraints += [cp.sum(self.weight_param) == total_weight]
        return constraints

    def leverage(self, leverage):
        return [cp.norm(self.weight_param, 1) <= leverage]

    def num_assets(self, num_assets):
        if self.ret_vec.shape[0] <= num_assets:
            warnings.warn("""The number of assets to hold exceeds the number of assets available, 
            default to a 1 asset only scenario""")
            num_assets = self.ret_vec.shape[0] - 1
        return [cp.sum_smallest(cp.abs(self.weight_param), self.ret_vec.shape[0] - num_assets) <= 0.0001]

    def concentration(self, top_holdings, top_concentration):
        if self.ret_vec.shape[0] <= top_holdings:
            warnings.warn("""Number of Top Holdings Exceeds Total Available Assets. 
            Will default top_holdings to be number of holdings available""")
            top_holdings = self.ret_vec.shape[0]
        return [cp.sum_largest(cp.norm(self.weight_param, 1), top_holdings) <= top_concentration]


    ### Market Data Needed/Calculation Needed
    def market_neutral(self, bound):

        market_cap_weight = self.market_cap_data()
        return [market_cap_weight @ self.moment_mat @ self.weight_param >= bound[0],
                market_cap_weight @ self.moment_mat @ self.weight_param <= bound[1]]

    # Return related
    def expected_return_const(self, bound, time_scaling):
        bound = self.construct_bound(bound, True, np.inf)
        return [cp.power(1 + self.expected_return(), time_scaling) - 1 >= bound[0], cp.power(1 + self.expected_return(), time_scaling) - 1 <= bound[1]]

    def sharpe_const(self, risk_free, bound):
        bound = self.construct_bound(bound, True, np.inf)
        return [self.sharpe(risk_free) >= bound[0], self.sharpe(risk_free) <= bound[1]]

    def beta_const(self, bound, individual_beta):
        bound = self.construct_bound(bound, False, -0.1)
        return [self.beta(individual_beta) >= bound[0], self.beta(individual_beta) <= bound[1]]

    def treynor_const(self, bound, risk_free, individual_beta):
        bound = self.construct_bound(bound, True, np.inf)
        return [self.treynor(risk_free, individual_beta) >= bound[0], self.treynor(risk_free, individual_beta) <= bound[1]]

    def jenson_alpha_const(self, bound, risk_free, market_return, individual_beta):
        bound = self.construct_bound(bound, True, np.inf)
        return [self.jenson_alpha(risk_free, market_return, individual_beta) >= bound[0], self.jenson_alpha(risk_free, market_return, individual_beta) <= bound[1]]

    # Risk related constraints
    def volatility_const(self, bound):
        return [self.volatility() >= bound[0], self.volatility <= bound[1]]

    def variance_const(self, bound):
        if self.moment != 2:
            raise DimException("Did not pass in a correlation/covariance matrix")
        return [self.variance() >= bound[0], self.variance() <= bound[1]]

    def skew_const(self, bound):
        if self.moment != 3:
            raise DimException("Did not pass in a coskewness matrix")
        return [self.higher_moment(3) >= bound[0], self.higher_moment(3) <= bound[1]]

    def kurt_const(self, bound):
        if self.moment != 4:
            raise DimException("Did not pass in a cokurtosis matrix")
        return [self.higher_moment(4) >= bound[0], self.higher_moment(4) <= bound[1]]

    def moment_const(self, bound):
        return [self.higher_moment(self.moment) >= bound[0], self.higher_moment(self.moment) <= bound[1]]


    def construct_bound(self, bound, minimum, opposite_value):
        if isinstance(bound, (int, float)):
            warnings.warn(f"""Only one bound is given, will set the {'maximum' if minimum else 'minimum'} value to be {opposite_value}""")
            if minimum:
                bound = (bound, opposite_value)
            else:
                bound = (opposite_value, bound)
        return bound




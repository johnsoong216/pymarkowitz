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
        # self.weight_param = weight_param
        # self.ret_vec = ret_vec
        # self.moment_mat = moment_mat
        # self.moment = moment

        self.method_dict = {"min_variance": self.min_variance}

    def create_objective(self, objective_type, **kwargs):
        return self.method_dict[objective_type](**kwargs)

    def equal_risk_parity(self):
        return cp.Minimize(self.risk_parity())

    def min_correlation(self):
        return cp.Minimize(self.correlation())

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

    @staticmethod
    def min_correlation():
        pass

    @staticmethod
    def equal_risk_contribution():
        pass

    @staticmethod
    def inv_volatility():
        pass

    @staticmethod
    def inv_variance():
        pass

    @staticmethod
    def equal_weight():
        pass

    @staticmethod
    def min_skew():
        pass

    @staticmethod
    def min_kurt():
        pass

    @staticmethod
    def min_moment():
        pass



    # def __init__(self, ret_data, moment_data, moment):
    #     self.ret_data = ret_data
    #     self.moment_data = moment_data
    #     self.moment = moment
    #     self.constraints = []

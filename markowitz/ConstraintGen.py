import numpy as np
import pandas as pd
import cvxpy as cp
import math
import warnings
from .Exceptions import *

### https://www.portfolioprobe.com/features/constraints/


class ConstraintGen:

    ### Weight Only
    @staticmethod
    def weight():
        pass

    @staticmethod
    def leverage():
        pass

    @staticmethod
    def num_assets():
        pass

    @staticmethod
    def concentration():
        pass

    ### Market Data Needed/Calculation Needed
    @staticmethod
    def market_neutral():
        pass

    @staticmethod
    def beta():
        pass

    @staticmethod
    def risk_fraction():
        pass

    @staticmethod
    def sortino():
        pass

    @staticmethod
    def calmar():
        pass

    @staticmethod
    def omega():
        pass

    @staticmethod
    def treynor():
        pass

    @staticmethod
    def sharpe():
        pass

    @staticmethod
    def sharpe():
        pass

    @staticmethod
    def tracking_error():
        pass

    @staticmethod
    def volatility():
        pass

    @staticmethod
    def skewness():
        pass

    @staticmethod
    def kurtosis():
        pass

    @staticmethod
    def higher_moment():
        pass

    @staticmethod
    def correlation():
        pass

    @staticmethod
    def expected_return():
        pass







    # def __init__(self, ret_data, moment_data, moment, weight_params):
    #
    #     self.ret_data = ret_data
    #     self.moment_data = moment_data
    #     self.moment = moment
    #     self.weight_params = weight_params
    #
    #     self.objective = None
    #     self.constraints = []
    #
    #     self.constraint_dict = {}
    #     self.objective_dict = {}
    #
    #
    # def add_objective(self, obj_type, **kwargs):
    #     self.objective = self.objective_dict[obj_type](**kwargs)
    #
    # def add_constraint(self, const_type, **kwargs):
    #     self.constraints += self.constraint_dict[const_type](**kwargs)










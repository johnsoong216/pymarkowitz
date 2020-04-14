import numpy as np
import pandas as pd
import cvxpy as cp
import math
import warnings
from .Exceptions import *


class ObjectiveGen:

    @staticmethod
    def min_volatility():
        pass

    @staticmethod
    def max_diversification():
        pass

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

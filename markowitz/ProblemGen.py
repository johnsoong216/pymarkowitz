import numpy as np
import pandas as pd
import cvxpy as cp
import math
import warnings
from .Exceptions import *



class ProblemGen:

    def __init__(self, ret_data, moment_data, asset_names=None):

        self.ret_vec, self.moment_mat, self.assets, self.moment = ProblemGen.init_checker(ret_data, moment_data,
                                                                                         asset_names)
        self.weight_params = cp.Variable(self.ret_vec.shape[0])
        self.weight_sols = None

        self.objective = None
        self.constraints = []
        self.objective_dict = {}
        self.constraint_dict = {}

    ### Add some quick shortcuts

    def add_objective(self, objective_type, **kwargs):
        self.objective = self.objective_dict[objective_type](**kwargs)

    def add_constraint(self, constraint_type, **kwargs):
        self.constraints += [self.constraint_dict[constraint_type](**kwargs)]

    def clear_problem(self, clear_obj=True, clear_constraints=True):

        self.weight_params = cp.Variable(self.ret_vec.shape[0])
        self.weight_sols = None

        if clear_constraints:
            self.constraints = []
        if clear_obj:
            self.objective = None

    def solve_problem(self, minimize):

        if minimize:
            prob = cp.Problem(cp.Minimize(self.objective), self.constraints)
        else:
            prob = cp.Problem(cp.Maximize(self.objective), self.constraints)

        if not (prob.is_dcp() and prob.is_dgp()):
            raise OptimizeException(f"""The problem formulated is not {'convex' if minimize else 'concave'}""")

        prob.solve()

        if "unbounded" in prob.status:
            raise OptimizeException("Unbounded Variables")
        elif "infeasible" in prob.status:
            raise OptimizeException("Infeasible Variables")
        elif "inaccurate" in prob.status:
            warnings.warn("Results may be inaccurate.")

        self.weight_sols = self.weight_params.value

    @staticmethod
    def init_checker(ret_data, moment_data, asset_names):

        asset_candidates = None
        if isinstance(ret_data, pd.Series):
            ret_vec = ret_data.values
            asset_candidates = list(ret_data.index)
        elif isinstance(ret_data, list):
            ret_vec = np.array(ret_data)
        elif isinstance(ret_data, np.ndarray):
            ret_vec = ret_data.reshape(-1)
        else:
            raise FormatException("""Return Vector must be a pd.Series, list or np.ndarray object""")

        if isinstance(moment_data, pd.DataFrame):
            moment_mat = moment_data.values.T
            asset_candidates = list(moment_data.index)
        elif isinstance(moment_data, np.ndarray):
            moment_mat = moment_data
        else:
            raise FormatException("""Moment Matrix must be a pd.DataFrame or np.ndarray object""")

        moment = math.log(moment_mat.shape[1], moment_mat.shape[0])

        if asset_names:
            assets = asset_names
        elif asset_candidates:
            assets = asset_candidates
        else:
            assets = [f'ASSET_{x}' for x in range(moment_mat.shape[0])]

        ### Dimensionality Checking
        if ret_vec.shape[0] != moment_mat.shape[0]:
            raise DimException("""Inconsistent Shape between Return Vector and the Moment Matrix""")
        elif int(moment) != moment:
            raise DimException("""Incorrect Dimension of the Moment Matrix""")

        return ret_vec, moment_mat, assets, moment





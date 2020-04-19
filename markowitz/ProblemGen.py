
import math
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize

from .Exceptions import *
from .ConstraintGen import ConstraintGen as ConstGen
from .ObjectiveGen import ObjectiveGen as ObjGen



class ProblemGen:

    def __init__(self, ret_data, moment_data, asset_names=None):

        self.ret_vec, self.moment_mat, self.assets, self.moment = ProblemGen.init_checker(ret_data, moment_data,
                                                                                         asset_names)
        # self.weight_params = cp.Variable(self.ret_vec.shape[0])
        # self.weight_params =
        self.weight_params = None
        self.weight_sols = None

        self.objective = None
        self.objective_sol = None
        self.objective_args = None
        self.constraints = []
        self.bounds = list(zip(np.repeat(0, self.ret_vec.shape[0]), np.repeat(1, self.ret_vec.shape[0])))

        self.obj_creator = ObjGen(self.weight_params, self.ret_vec, self.moment_mat, self.moment, self.assets)
        self.const_creator = ConstGen(self.weight_params, self.ret_vec, self.moment_mat, self.moment, self.assets)

    ### Add some quick shortcuts

    def add_objective(self, objective_type, **kwargs):

        self.objective = self.obj_creator.create_objective(objective_type)
        self.objective_args = tuple(kwargs.values())

    def add_constraint(self, constraint_type, **kwargs):
        if constraint_type == "weight":
            bound, total = self.const_creator.create_constraint(constraint_type, **kwargs)
            # print(bound)
            # print(total)
            self.bounds = bound
            self.constraints += total
        else:
            self.constraints += self.const_creator.create_constraint(constraint_type, **kwargs)

    def clear(self, clear_obj=True, clear_constraints=True):

        self.weight_params = cp.Variable(self.ret_vec.shape[0])
        self.weight_sols = None

        if clear_constraints:
            self.constraints = []
        if clear_obj:
            self.objective = None

    def solve(self):
        if type(self.objective) != np.ndarray:
            # print(self.bounds)
            # print(self.bounds)
            # print(self.constraints)
            res = minimize(self.objective, x0=np.random.uniform(size=self.ret_vec.shape[0]),
                           constraints=self.constraints, bounds=self.bounds, args=self.objective_args)
            # if not res.success:
            #     raise OptimizeException("""Optimization has failed. Please adjust parameters""")
            print(res)
            # try:
            #     ans = prob.solve()
            # except cp.DCPError:
            #     try:
            #         ans = prob.solve(qcp=True)
            #     except (cp.DCPError, cp.SolverError):
            #         try:
            #             ans = prob.solve(solver=cp.SCS, qcp=True)
            #         except cp.DCPError:
            #             raise OptimizeException(f"""The problem formulated is not convex if minimizing,
            #         concave if maximizing""")
            #
            # if "unbounded" in prob.status:
            #     raise OptimizeException("Unbounded Variables")
            # elif "infeasible" in prob.status:
            #     raise OptimizeException("Infeasible Variables")
            # elif "inaccurate" in prob.status:
            #     warnings.warn("Results may be inaccurate.")

            self.weight_sols = res.x
            # self.weight_sols = dict(zip(self.assets, self.obj_creator.weight_param.value.round(5)))
            # self.weight_sols_2 = dict(zip(self.assets, self.const_creator.weight_param.value.round(5)))
        else:
            warnings.warn(f"""The problem formulated is not an optimization problem and is calculated numerically""")
            self.weight_sols = dict(zip(self.assets, self.objective))

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

        moment = math.log(moment_mat.shape[1], moment_mat.shape[0]) + 1

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





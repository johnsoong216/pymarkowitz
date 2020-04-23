"""
Optimizer Class Constructs Mean-Variance Related Optimization Problems with Constraints

2 Major Functionality:
- Optimize Weight based on Constraints & Objectives
- Simulate Random Weight Scenarios

For the first functionality, all the addition of objective/constraints are performed with the following methods.
- add_objective()
- add_constraint()

For the second functionality, all the weight-related constraints can be passed in as arguments in the following method:
- simulate()
"""

import math
import warnings
import inspect
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from .Exceptions import *
from .Constraints import ConstraintGenerator as ConstGen
from .Objectives import ObjectiveGenerator as ObjGen
from .Metrics import MetricGenerator as MetGen


class Optimizer:

    def __init__(self, ret_data, moment_data, beta_data=None, asset_names=None):
        """
        Initializes an Optimizer instance with data

        Default constraints are: weight bound is (0,1), Leverage is 1 (Long only portfolio without additional margin)

        :param ret_data: pd.DataFrame/np.ndarray, return data
        :param moment_data: pd.DataFrame/np.ndarray, moment data (covariance/coskewness/... matrix)
        :param beta_data: pd.Series/np.ndarray, optional, beta data for each asset
                If not provided then beta related constraints/objectives can not be optimized/simulated
        :param asset_names: List[str], optional, list of asset names
        """

        self.ret_vec, self.moment_mat, self.assets, self.moment, self.beta_vec = Optimizer.init_checker(ret_data, moment_data,
                                                                                                        asset_names, beta_data)

        self.weight_sols = None

        self.objective = None
        self.objective_sol = None
        self.objective_args = None

        self.obj_creator = ObjGen(self.ret_vec, self.moment_mat, self.moment, self.assets, self.beta_vec)
        self.const_creator = ConstGen(self.ret_vec, self.moment_mat, self.moment, self.assets, self.beta_vec)
        self.metric_creator = MetGen(self.ret_vec, self.moment_mat, self.moment, self.assets, self.beta_vec)

        self.bounds, self.constraints = self.const_creator.create_constraint('weight', weight_bound=(0, 1), leverage=1)
        self.leverage = 1

    def add_objective(self, objective_type, **kwargs):
        """
        Add an objective to the optimization problem. Call objective_options() to check all available options.
        You can also input a customized objective by setting objective_type="custom".
        The custom_func should follow the parameter structure of custom_func(w, **kwargs)

        The limitation with custom_func is it cannot call the moment matrix/return matrix/beta vector that are passed into MetricGenerator
        :param objective_type: str, objective name
        :param kwargs: arguments to be passed into the objective when performing optimization
        """
        if objective_type != "custom":
            self.objective_args = tuple(kwargs.values())
            self.objective = self.obj_creator.create_objective(objective_type, **kwargs)
        else:
            self.objective_args = tuple(kwargs.values())[1:]
            self.objective = tuple(kwargs.values())[0]

    def add_constraint(self, constraint_type, **kwargs):
        """
        Add an objective to the optimization problem. Call constraint_options() to check all available options.
        You can also input a customized objective by setting constraint_type="custom".
        The custom_func should follow the parameter structure of custom_func(w, **kwargs)

        The limitation with custom_func is it cannot call the moment matrix/return matrix/beta vector that are passed into MetricGenerator
        :param objective_type: str, objective name
        :param kwargs: arguments to be passed into the constraints
        """
        if constraint_type == "custom":
            self.constraints += tuple(kwargs.values())[0]
        elif constraint_type == "weight":
            bound, leverage = self.const_creator.create_constraint(constraint_type, **kwargs)
            self.bounds = bound
            self.leverage = kwargs['leverage']
            self.constraints[0] = leverage[0] # Total Leverage is always the first constraint
        else:
            self.constraints += self.const_creator.create_constraint(constraint_type, **kwargs)

    def clear(self, clear_obj=True, clear_constraints=True):
        """
        Clear the optimization problem
        :param clear_obj: bool, Clear the objective
        :param clear_constraints: bool, clear the constraints. Note that weight and leverage will be defaulted to (0,1) and leverage of 1 after clearance
        """

        if clear_constraints:
            self.constraints = []
            self.bounds, self.constraints = self.const_creator.create_constraint('weight', weight_bound=(0,1), leverage=1)
        if clear_obj:
            self.objective = None

    def solve(self, x0=None, round_digit=4, **kwargs):
        """
        Solves the optimization problem
        :param x0: np.ndarray, default=None User can pass in an initial guess to avoid scipy from running into local minima
        :param round_digit: int, default=4, round portfolio weight
        :param kwargs: arguments for method clear()
        """
        if type(self.objective) != np.ndarray:
            res = minimize(self.objective, x0 = ConstGen.gen_random_weight(self.ret_vec.shape[0], self.bounds, self.leverage) if x0 is None else x0, options={'maxiter': 1000},
                           constraints=self.constraints, bounds=self.bounds, args=self.objective_args)
            if not res.success:
                self.clear(**kwargs)
                raise OptimizeException(f"""Optimization has failed. Error Message: {res.message}. 
                                            Please adjust constraints/objectives or input an initial guess.""")

            self.clear(**kwargs)
            self.weight_sols = np.round(res.x, round_digit) + 0

        else:
            warnings.warn(f"""The problem formulated is not an optimization problem and is calculated numerically""")

            self.weight_sols = np.round(self.objective, round_digit) + 0
            self.clear(**kwargs)

    def summary(self, risk_free=None, market_return=None, top_holdings=None, round_digit=4):
        """
        Returns a tuple of dictionaries - Weight dictionary, Metrics Dictionary
        :param risk_free: float, default=None, if pass in a float can compute additional metrics in summary
        :param market_return: float, default=None, if pass in a float can compute additional materics in summary
        :param top_holdings: int, default=None, number of holdings, if pass in can compute additional metrics in summary
        :param round_digit: int, round the metrics to the xth decimal place
        :return: tuple[dict]
        """

        moment_dict = defaultdict(lambda: "Moment")
        moment_dict[3] = "Skewness"
        moment_dict[4] = "Kurtosis"

        weight_dict = dict(zip(self.assets, self.weight_sols))
        metric_dict = {'Expected Return': self.metric_creator.expected_return(self.weight_sols),
                       "Leverage": self.metric_creator.leverage(self.weight_sols),
                       "Number of Holdings": self.metric_creator.num_assets(self.weight_sols)}

        # Portfolio Composition
        if top_holdings:
            metric_dict[f"Top {top_holdings} Holdings Concentrations"] = self.metric_creator.concentration(
                self.weight_sols, top_holdings)

        # Risk Only
        if self.moment == 2:
            metric_dict["Volatility"] = self.metric_creator.volatility(self.weight_sols)
            # metric_dict["Correlation"] = self.metric_creator.correlation(self.weight_sols)
        else:
            metric_dict[f'{moment_dict[int(self.moment)]}'] = self.metric_creator.higher_moment(self.weight_sols)

        # Risk-Reward
        if self.beta_vec is not None:
            metric_dict["Portfolio Beta"] = self.metric_creator.beta(self.weight_sols)

        if risk_free is not None:
            metric_dict["Sharpe Ratio"] = self.metric_creator.sharpe(self.weight_sols, risk_free)

        if self.beta_vec is not None and risk_free is not None:
            metric_dict["Treynor Ratio"] = self.metric_creator.treynor(self.weight_sols, risk_free)
            if market_return is not None:
                metric_dict["Jenson's Alpha"] = self.metric_creator.jenson_alpha(self.weight_sols, risk_free, market_return)

        for item in metric_dict:
            metric_dict[item] = np.round(metric_dict[item], round_digit)

        weight_dict = {k: v for k, v in weight_dict.items() if v}
        return weight_dict, metric_dict

    def simulate(self, x='volatility', y='expected_return', iters=1000, weight_bound=(0,1), leverage=1, ret_format='df',
                 file_path=None, x_var=None, y_var=None):
        """
        Simulate random weight scenarios with flexible x/y variables.
        Call metric_options() to see all possible x,y combinations and their respective signature

        :param x: str, name of metric 1. If returning a plot will be the x-axis metric
        :param y: str, name of metric 2. If returning a plot will be the y-axis metric
        :param iters: int, number of simulations
        :param weight_bound: tuple/np.ndarray/List[tuple], weight bound
        :param leverage: float, total leverage
        :param ret_format: str, default='df', additional options ["plotly", "sns"
                            If selected sns will return a plt figure
                            If selected plotly will return a plotly.express figure
                            If selected df will return a dataframe with x,y, and weight values
        :param file_path: str, default=None, path for saving plt figure
        :param x_var: dict, optional. Additional parameters needed to compute x
        :param y_var: dict, optional. Additional parmaeters needed to compute y
        :return:
        """
        if y_var is None:
            y_var = dict()
        if x_var is None:
            x_var = dict()

        x_val = np.zeros(iters)
        y_val = np.zeros(iters)
        weight_vals = np.zeros(shape=(iters, len(self.assets)))
        individual_bound = ConstGen.construct_weight_bound(self.ret_vec.shape[0], (0,1), weight_bound)

        for it in range(iters):
            temp_weights = ConstGen.gen_random_weight(self.ret_vec.shape[0], individual_bound, leverage)
            weight_vals[it] = temp_weights
            x_val[it] = self.metric_creator.method_dict[x](temp_weights, **x_var)
            y_val[it] = self.metric_creator.method_dict[y](temp_weights, **y_var)

        if ret_format == 'sns': # Change to plt, fig format
            fig, ax = plt.subplots(figsize=(18, 12));
            ax = sns.scatterplot(x_val, y_val);
            ax.set_title(f"{x} VS {y}")
            plt.xlim(x_val.min(), x_val.max());
            plt.ylim(y_val.min(), y_val.max());
            plt.xlabel(x);
            plt.ylabel(y);
            if file_path:
                plt.savefig(file_path)
            plt.show()
        else:
            res_df = pd.DataFrame(columns=[x] + [y], data=np.concatenate([x_val.reshape(1, -1), y_val.reshape(1, -1)]).T)
            res_df = pd.concat([res_df, pd.DataFrame(columns=self.assets, data=weight_vals)], axis=1)
            if ret_format == 'plotly':
                return px.scatter(res_df, x=x, y=y, title=f"{x} vs {y}")
            elif ret_format == "df":
                return res_df
            else:
                raise FormatException("""Return Format must be sns, plotly, df""")

    def simulate_efficient_frontier(self, iters=1000, weight_bound=(0,1), leverage=1, num_assets=None, top_holdings=None, top_concentration=None, ret_format='df', file_path=None):
        """
        Simulate the efficient frontier (Quadratic Utility Function concerned with Expected Return and Variance Tradeoff

        :param iters: number of simulations
        :param weight_bound: constraints on individual portfolio
        :param leverage: constraint on total leverage
        :param num_assets: constraint on number of assets
        :param top_holdings: constraint on portfolio concentration
        :param top_concentration: constraint on portfolio concentration
        :param ret_format: return format
        :param file_path: save figure or not
        :return:
        """
        x_val = np.zeros(iters)
        y_val = np.zeros(iters)
        weight_vals = np.zeros(shape=(iters, len(self.assets)))
        self.bounds, self.constraints = self.const_creator.create_constraint('weight', weight_bound=weight_bound,
                                                                             leverage=leverage)
        if num_assets is not None:
            self.constraints += self.const_creator.create_constraint("num_assets", num_assets=num_assets)
        if top_holdings is not None and top_concentration is not None:
            self.constraints += self.const_creator.create_constraint("concentration", top_holdings=top_holdings,
                                                                     top_concentration=top_concentration)

        aversion_factor = np.logspace(-3, 3, iters)
        for it, av in enumerate(aversion_factor):
            self.objective = self.obj_creator.create_objective('efficient_frontier')
            self.objective_args = av
            self.solve(clear_constraints=False)
            x_val[it] = self.metric_creator.method_dict['volatility'](self.weight_sols)
            y_val[it] = self.metric_creator.method_dict['expected_return'](self.weight_sols)
            weight_vals[it] = self.weight_sols

        if ret_format == 'sns':  # Change to plt, fig format
            fig, ax = plt.subplots(figsize=(18, 12));
            ax = sns.scatterplot(x_val, y_val);
            ax.set_title(f"Efficient Frontier")
            plt.xlim(0, x_val.mean() + 3 * x_val.std());
            plt.ylim(y_val.min() - 0.01, y_val.max() + 0.01);
            plt.xlabel('volatility');
            plt.ylabel('expected_return');
            if file_path:
                plt.savefig(file_path)
            plt.show()
        else:
            res_df = pd.DataFrame(columns=['volatility', 'expected_return'],
                                  data=np.concatenate([x_val.reshape(1, -1), y_val.reshape(1, -1)]).T)
            res_df = pd.concat([res_df, pd.DataFrame(columns=self.assets, data=weight_vals)], axis=1)
            if ret_format == 'plotly':
                return px.scatter(res_df, x='volatility', y='expected_return', title=f"Efficient Frontier")
            elif ret_format == "df":
                return res_df
            else:
                raise FormatException("""Return Format must be sns, plotly, df""")

    def objective_options(self):
        """
        Returns a dictionary of objective options
        :return:
        """
        return Optimizer.list_method_options(self.obj_creator.method_dict)

    def constraint_options(self):
        """
        Returns a dictionary of constraint options
        :return:
        """
        return Optimizer.list_method_options(self.const_creator.method_dict)

    def metric_options(self):
        """
        Returns a dictionary of metric options
        :return:
        """
        return Optimizer.list_method_options(self.metric_creator.method_dict)

    @staticmethod
    def list_method_options(method_dict):
        """
        List all method options
        """
        res_dict = {}
        for method in method_dict:
            res_dict[method] = inspect.signature(method_dict[method])
        return res_dict

    @staticmethod
    def init_checker(ret_data, moment_data, asset_names, beta_data):
        """
        Dimensionality check when initializing Optimizer
        """

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
            moment_mat = moment_data.values
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

        beta_vec = None
        if beta_data is None:
            warnings.warn(""""Detected no beta input. Will not be able to perform any beta-related optimization.""")
        elif isinstance(beta_data, np.ndarray):
            warnings.warn(f"""Assume that beta input is in the sequence of {assets}.""")
        elif isinstance(beta_data, pd.Series):
            if list(beta_data.index) != assets:
                raise DimException(f"""Beta data must include all assets: {assets}""")
            else:
                beta_vec = beta_data[assets].values
        elif len(assets) != beta_data.shape[0]:
            raise DimException("""Inconsistent Shape between Beta Vector and the number of assets""")
        else:
            raise FormatException(f"""Beta data must be passed in as np.ndarray or pd.Series""")

        if ret_vec.shape[0] != moment_mat.shape[0]:
            raise DimException("""Inconsistent Shape between Return Vector and the Moment Matrix""")
        elif int(moment) != moment:
            raise DimException("""Incorrect Dimension of the Moment Matrix""")

        return ret_vec, moment_mat, assets, int(moment), beta_vec





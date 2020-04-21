"""
ConstraintGenerator constructs 3 types of optimization constraints
    - Portfolio Composition (Weight, Leverage, Concentration, Number of Holdings)
    - Risk-only (Volatility, Variance, Skewness, Kurtosis, Higher Normalized Moments, Market Neutral)
    - Risk Reward (Expected Return, Sharpe, Sortino, Beta, Treynor, Jenson's Alpha)

In addition, the class holds functions to construct bounds for constraints and weights
"""
import warnings

from .Metrics import *


class ConstraintGenerator(MetricGenerator):

    def __init__(self, ret_vec, moment_mat, moment, assets, beta_vec):
        """
        Initialize a ConstraintGenerator class with parameters to construct constraints
        Parameters are identical to its parent class MetricGenerator
        """

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
                            "variance": self.moment_const,
                            "skew": self.moment_const,
                            "kurt": self.moment_const,
                            "moment": self.moment_const}

    def create_constraint(self, constraint_type, **kwargs):
        """
        Universal method for creating a constraint
        :param constraint_type: str, options are listed in ConstraintGenerator.method_dict
        :param kwargs: arguments to be passed in to construct constraints
        :return: List[dict]
        """
        return self.method_dict[constraint_type](**kwargs)

    # Portfolio Composition
    def weight(self, weight_bound, leverage):
        """
        Constructing individual portfolio weight bound and total leverage

        :param weight_bound: np.ndarray/List/Tuple
                User can pass in a weight bound that universally applies to all individual assets Ex. weight_bound=(0,1)
                Also, user can specify each individual asset's weight bound by passing in a list of tuples/an np.ndarray
        :param leverage: float, the total leverage constraint for the portfolio
        :return: tuple of individual bound and total leverage functions
        """
        init_bound = (0,1)
        individual_bound = ConstraintGenerator.construct_weight_bound(self.ret_vec.shape[0], init_bound, weight_bound)

        total_leverage = [{'type': 'eq', 'fun': lambda w: -self.leverage(w) + leverage}]
        return individual_bound, total_leverage

    def num_assets_const(self, num_assets):
        """
        Constraint on the number of assets that can be held
        :param num_assets: int, number of assets
        :return: List[dict]
        """

        if self.ret_vec.shape[0] <= num_assets:
            warnings.warn("""The number of assets to hold exceeds the number of assets available, 
            default to a 1 asset only scenario""")
            num_assets = self.ret_vec.shape[0] - 1
        non_holdings = self.ret_vec.shape[0] - num_assets
        return [{'type': 'eq', 'fun': lambda w: np.sum(np.partition(np.sqrt(np.square(w)), non_holdings)[:non_holdings])}]

    def concentration_const(self, top_holdings, top_concentration):
        """
        Constraint on the concentration of the portfolio in the most heavily weighted assets
        :param top_holdings: int, number of top holdings to calculate concentration
        :param top_concentration: float, the maximum % concentration of the top holdings
        :return: List[dict]
        """
        if self.ret_vec.shape[0] <= top_holdings:
            warnings.warn("""Number of Top Holdings Exceeds Total Available Assets. 
            Will default top_holdings to be number of holdings available""")
            top_holdings = self.ret_vec.shape[0]

        return [{"type": "ineq", "fun": lambda w: np.sum(
            np.partition(-np.sqrt(np.square(w)), top_holdings)[:top_holdings]) / np.sum(
            np.sqrt(np.square(w))) + top_concentration}]

    # Risk Only
    # def market_neutral_const(self, bound):
    #     """
    #     Market neutral constraint. Ensures that the market neutral risk (based on market cap weight falls within)
    #     :param bound: tuple
    #     :return:
    #     """
    #     market_cap_weight = MetricGenerator.market_cap_data(self.assets)
    #     return [{"type": "ineq", "fun": lambda w: market_cap_weight @ self.moment_mat @ w - bound[0]},
    #             {"type": "ineq", "fun": lambda w: -(market_cap_weight @ self.moment_mat @ w - bound[1])}]

    # Risk related constraints
    def volatility_const(self, bound):
        """
        Constraint on portfolio volatility
        :param bound: float/tuple,
                If passed in tuple, then construct lower bound and upper bound
                Otherwise, assume passed in an upper bound
        :return:List[dict]
        """
        bound = ConstraintGenerator.construct_const_bound(bound, False, 0)
        return [{"type": "ineq", "fun": lambda w: self.volatility(w)  + bound[0]},
                {"type": "ineq", "fun": lambda w: -self.volatility(w) + bound[1]}]

    # Risk-Reward related constraints
    def expected_return_const(self, bound):
        """
        Constraint on expected return
        :param bound: float/tuple,
                If passed in tuple, then construct lower bound and upper bound
                Otherwise, assume passed in a lower bound
        :return:List[dict]
        """
        bound = ConstraintGenerator.construct_const_bound(bound, True, 10)
        return [{"type": "ineq", "fun": lambda w: self.expected_return(w) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.expected_return(w) + bound[1]}]

    def sharpe_const(self, risk_free, bound):
        """
        Constraint on sharpe ratio
        :param bound: float/tuple,
                If passed in tuple, then construct lower bound and upper bound
                Otherwise, assume passed in a lower bound
        :return:List[dict]
        """
        bound = ConstraintGenerator.construct_const_bound(bound, True, 10)

        return [{"type": "ineq", "fun": lambda w: self.sharpe(w, risk_free) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.sharpe(w, risk_free) + bound[1]}]

    def beta_const(self, bound):
        """
        Constraint on portfolio beta
        :param bound: float/tuple,
                If passed in tuple, then construct lower bound and upper bound
                Otherwise, assume passed in an upper bound
        :return:List[dict]
        """
        bound = ConstraintGenerator.construct_const_bound(bound, False, -1)
        return [{"type": "ineq", "fun": lambda w: self.beta(w) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.beta(w) + bound[1]}]

    def treynor_const(self, bound, risk_free):
        """
        Constraint on treynor
        :param bound: float/tuple,
                If passed in tuple, then construct lower bound and upper bound
                Otherwise, assume passed in a lower bound
        :param risk_free: int, risk free rate of return
        :return: List[dict]
        """
        bound = ConstraintGenerator.construct_const_bound(bound, True, 10)
        return [{"type": "ineq", "fun": lambda w: self.treynor(w, risk_free) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.treynor(w, risk_free)  + bound[1]}]

    def jenson_alpha_const(self, bound, risk_free, market_return):
        """
        Constraint on jenson's alpha
        :param bound: float/tuple,
                If passed in tuple, then construct lower bound and upper bound
                Otherwise, assume passed in a lower bound
        :param risk_free: float, risk free rate of return
        :param market_return: float, market return
        :return: List[dict]
        """
        bound = ConstraintGenerator.construct_const_bound(bound, True, 10)
        return [{"type": "ineq", "fun": lambda w: self.jenson_alpha(w, risk_free, market_return) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.jenson_alpha(w, risk_free, market_return) + bound[1]}]

    def moment_const(self, bound):
        """
        Constraint on moment (variance, skewness, kurtosis, higher moment)
        :param bound: float/tuple,
                If passed in tuple, then construct lower bound and upper bound
                Otherwise, assume passed in a lower bound
        :return: List[dict]
        """
        bound = ConstraintGenerator.construct_const_bound(bound, False, 0)
        return [{"type": "ineq", "fun": lambda w: self.higher_moment(w) + bound[0]},
                {"type": "ineq", "fun": lambda w: -self.higher_moment(w) + bound[1]}]
    
    @staticmethod
    def construct_const_bound(bound, minimum, opposite_value):
        """
        Constructing constraint bound based on input
        :param bound: int/float/tuple, bound value
        :param minimum: bool, indicate whether passed in parameter is upper/lower bound
        :param opposite_value: int/float, the opposite value of the bound
        :return: tuple
        """
        if isinstance(bound, (int, float)):
            warnings.warn(f"""Only one bound is given, will set the {'maximum' if minimum else 'minimum'} value to be {opposite_value}""")
            if minimum:
                bound = (bound, opposite_value)
            else:
                bound = (opposite_value, bound)
        return bound
    
    @staticmethod
    def construct_weight_bound(size, init_bound, weight_bound):
        """
        Construct portfolio weight bound
        :param size: int, number of assets
        :param init_bound: tuple, initial bound (0, 1)
        :param weight_bound: list/tuple/np.ndarray, user-determined constraint on individual weight
        :return: List[tuple] List of Tuples of Lower/Upper bounds
        """

        individual_bound = init_bound

        if isinstance(weight_bound, (list, tuple)):
            if isinstance(weight_bound[0], (list, tuple)):
                if all([len(ind_weights) == 2 for ind_weights in weight_bound]) and len(weight_bound) == size:
                    weight_bound = np.array(weight_bound)
                else:
                    raise DimException("""If specifying weight for each individual asset, must be passed in pairs and 
                                            its length must equal the number of assets""")
            if isinstance(weight_bound[0], (float, int)):
                # constraints += [self.weight_param >= weight_bound[0]]
                # constraints += [self.weight_param <= weight_bound[1]]
                individual_bound = list(zip(np.repeat(weight_bound[0], size),
                                            np.repeat(weight_bound[1], size)))
            else:
                raise FormatException(
                    """Please pass in weight boundaries in an accepted format. List/Tuple/Np.ndarray""")
        if isinstance(weight_bound, np.ndarray):
            if weight_bound.ndim == 1:
                individual_bound = list(zip(np.repeat(weight_bound[0], size),
                                            np.repeat(weight_bound[1], size)))

            elif weight_bound.ndim == 2:
                if weight_bound.shape != (size, 2):
                    raise DimException("Dimension of Weights does not match number of assets")
                individual_bound = list(zip(weight_bound[:, 0], weight_bound[:, 1]))
            else:
                raise DimException("Dimension of Weight Bound Array must be 1/2")
        return individual_bound
    
    @staticmethod
    def gen_random_weight(size, bound, leverage):
        """
        Generate Random Weights for Simulation
        :param size: int, number of portfolios
        :param bound: List[Tuple]
                If bounds are identical for every single asset than use dirichilet distribution to generate random
                portfolios. This has the advantage of creating highly concentrated/diversified portfolios that proxy
                real world portfolio allocations. Note that bound constraints are perfectly adhered to if leverage=1 and
                setting an extremely high leverage value may cause violations on bound constraints.
                If bounds are not identical then generate with normal distribution. Note that randomness deteriorates
                with more portfolios.
        :param leverage: float, total leverage
        :return: np.ndarray
        """
        if all(bound[0][0] == low for low, high in bound) and all(bound[0][1] == high for low, high in bound):
            rand_weight = np.random.dirichlet(np.arange(1, size + 1))
            if bound[0][0] < 0:
                neg_idx = np.random.choice(rand_weight.shape[0], np.random.choice(size + 1), replace=False)
                rand_weight[neg_idx] = -rand_weight[neg_idx]
                temp = rand_weight * (bound[0][1] - bound[0][0]) / 2 + (
                            bound[0][0] + (bound[0][1] - bound[0][0]) / 2)
            else:
                temp = rand_weight * (bound[0][1] - bound[0][0]) + bound[0][0]
        else:
            temp = np.zeros(shape=size)
            for idx, interval in enumerate(bound):
                val = np.random.randn(1)[0]
                std = (interval[1] - interval[0])/2
                mu = (interval[1] + interval[0])/2
                temp[idx] = val * std + mu

        temp = temp / np.abs(temp).sum() * leverage  # Two Standard Deviation
        return temp


    # def variance_const(self, bound):
    #     if self.moment != 2:
    #         raise DimException("Did not pass in a correlation/covariance matrix")
    #     bound = ConstraintGenerator.construct_const_bound(bound, False, 0)
    #     return [{"type": "ineq", "fun": lambda w: self.variance(w) + bound[0]},
    #             {"type": "ineq", "fun": lambda w: -self.variance(w) + bound[1]}]
    #
    # def skew_const(self, bound):
    #     if self.moment != 3:
    #         raise DimException("Did not pass in a coskewness matrix")
    #     bound = ConstraintGenerator.construct_const_bound(bound, False, 0)
    #     return [{"type": "ineq", "fun": lambda w: self.higher_moment(w) + bound[0]},
    #             {"type": "ineq", "fun": lambda w: -self.higher_moment(w) + bound[1]}]
    #
    # def kurt_const(self, bound):
    #     if self.moment != 4:
    #         raise DimException("Did not pass in a cokurtosis matrix")
    #
    #     bound = ConstraintGenerator.construct_const_bound(bound, False, 0)
    #     return [{"type": "ineq", "fun": lambda w: self.higher_moment(w) + bound[0]},
    #             {"type": "ineq", "fun": lambda w: -self.higher_moment(w) + bound[1]}]


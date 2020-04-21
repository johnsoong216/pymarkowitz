"""
ObjectiveGenerator constructs 3 types of optimization objectives
    - Risk-only (Volatility, Variance, Skewness, Kurtosis, Higher Normalized Moments, Market Neutral)
    - Risk Reward (Expected Return, Efficient Frontier, Sharpe, Sortino, Beta, Treynor, Jenson's Alpha)
    - Numerical (Inverse Volatility, Variance, Equal Weight, Market Cap Weight)
"""
import numpy as np
from .Metrics import MetricGenerator


class ObjectiveGenerator(MetricGenerator):

    def __init__(self, ret_vec, moment_mat, moment, assets, beta_vec):
        """
        Initialize an ObjectiveGenerator class with parameters to construct objectives
        Parameters are identical to its parent class MetricGenerator
        """

        super().__init__(ret_vec, moment_mat, moment, assets, beta_vec)

        self.method_dict = {"efficient_frontier": self.efficient_frontier,
                            "equal_risk_parity": self.equal_risk_parity,
                            "min_correlation": self.min_correlation,
                            "min_volatility": self.min_volatility,
                            "min_variance": self.min_moment,
                            "min_skew": self.min_moment,
                            "min_kurt": self.min_moment,
                            "min_moment": self.min_moment,
                            "max_return": self.max_return,
                            "max_diversification": self.max_diversification,
                            "max_sharpe": self.max_sharpe,
                            "min_beta": self.min_beta,
                            "max_treynor": self.max_treynor,
                            "max_jenson_alpha": self.max_jenson_alpha,
                            "inverse_volatility": self.inverse_volatility,
                            "inverse_variance": self.inverse_variance,
                            "equal_weight": self.equal_weight,
                            "market_cap_weight": self.market_cap_weight}

    def create_objective(self, objective_type, **kwargs):
        """
        Universal method for creating an objective
        :param objective_type: str, options are listed in ObjectiveGenerator.method_dict
        :param kwargs: arguments to be passed in to construct objectives
        :return: func/np.ndarray (if weight construction is purely numerical, then return the weight vector, else return a function)
        """
        if objective_type in ["equal_weight", "market_cap_weight", "inverse_volatility", "inverse_variance"]:
            return self.method_dict[objective_type](**kwargs)
        return self.method_dict[objective_type]

    # Risk Related
    def equal_risk_parity(self, w):
        """
        Objective: Individual Portfolios Contribute Equal amount of risk to the portfolio
        :param w: np.ndarray, weight vector
        :return: float
        """
        return self.risk_parity(w)

    def min_correlation(self, w):
        """
        Objective: Minimize Portfolio Correlation Factor
        :param w: np.ndarray, weight vector
        :return: float
        """
        return self.correlation(w)

    def min_volatility(self, w):
        """
        Objective: Minimize Portfolio Volatility
        :param w: np.ndarray, weight vector
        :return: float
        """
        return self.volatility(w)

    def min_moment(self, w):
        """
        Objective: Minimize Portfolio Moment (Variance if moment=2, Skewness if moment=3, Kurtosis if moment=4)
        :param w: np.ndarray, weight vector
        :return: float
        """
        return self.higher_moment(w)

    def max_diversification(self, w):
        """
        Objective: Maximize Portfolio Diversification Factor
        :param w: np.ndarray, weight vector
        :return: float
        """
        return -self.diversification(w)

    def efficient_frontier(self, w, aversion):
        """
        Objective: Maximize return with lowest variance (Classic Mean Variance Optimization)
        :param w: np.ndarray, weight vector
        :param aversion: float, risk aversion factor
        :return: float
        """
        return -(self.expected_return(w) - aversion * self.higher_moment(w))

    def max_return(self, w):
        """
        Objective: Maximize Return
        :param w: np.ndarray, weight vector
        :return: float
        """
        return -self.expected_return(w)

    def max_sharpe(self, w, risk_free):
        """
        Objective: Maximize Sharpe
        :param w: np.ndarray, weight vector
        :param risk_free: float, risk free rate of return. Must be positive
        :return: float
        """
        return -self.sharpe(w, risk_free)

    def min_beta(self, w):
        """
        Objective: Minimize Absolute Beta (Close to 0)
        :param w: np.ndarray, weight vector
        :return: float
        """
        return np.sqrt(np.square(self.beta(w)))

    def max_treynor(self, w, risk_free):
        """
        Objective: Maximize Treynor Ratio
        :param w: np.ndarray, weight vector
        :param risk_free: float, risk free rate of return
        :return: float
        """
        return -self.treynor(w, risk_free)

    def max_jenson_alpha(self, w, risk_free, market_return):
        """
        Objective: Maximizes Jenson's Alpha
        :param w: np.ndarray, weight vector
        :param risk_free: float, risk free rate of return
        :param market_return: float, assumed market rate of return
        :return: float
        """
        return -self.jenson_alpha(w, risk_free, market_return)



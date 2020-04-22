"""
MetricGenerator calculates common portfolio metrics for 3 functionalities:
    - To measure portfolio composition and performance
    - To use as constraints in optimization problem
    - To use as objectives for the optimization problem

Three Major Types of Metrics
    - Portfolio Composition: Leverage, Concentration, Number of Holdings
    - Risk-only: Correlation, Diversifcation Factor, Volatility, Variance, Skewness, Kurtosis, Higher Normalized Moments, Risk Parity
    - Risk-Reward: Expected Return, Sharpe, Sortino, Beta, Treynor, Jenson's Alpha


In addition, the class holds functions to calculate weight that do not require an optimization problem
    - inverse volatility/variance,
    - equal weight/market cap weight

Reference for Calculations of Correlation, Diversifcation & Risk Parity:
https://investresolve.com/file/pdf/Portfolio-Optimization-Whitepaper.pdf

Reference for Calculations for Sharpe, Sortino, Beta, Treynor, Jenson's Alpha
https://www.cfainstitute.org/-/media/documents/support/programs/investment-foundations/19-performance-evaluation.ashx?la=en&hash=F7FF3085AAFADE241B73403142AAE0BB1250B311
https://www.investopedia.com/terms/j/jensensmeasure.asp
https://www.investopedia.com/ask/answers/070615/what-formula-calculating-beta.asp
"""

import numpy as np
import pandas_datareader as data

from .Exceptions import *


class MetricGenerator:

    def __init__(self, ret_vec, moment_mat, moment, assets, beta_vec):

        """
        Initialize a MetricGenerator instance with parmaters to compute metrics

        :param ret_vec: np.ndarray, mean return vector
        :param moment_mat: np.ndarray, moment matrix (usually covariance) used for optimization
        :param moment: int, the type of moment matrix passed in, moment=2 -> covariance matrix
        :param assets: List[str], asset names, needed if pulling market cap data information
        :param beta_vec: np.ndarray, vector of beta of individual assets
        """

        self.ret_vec = ret_vec
        self.moment_mat = moment_mat
        self.moment = int(moment)
        self.assets = assets
        self.beta_vec = beta_vec

        self.method_dict = {"leverage": self.leverage,
                            "num_assets": self.num_assets,
                            "concentration": self.concentration,
                            "correlation": self.correlation,
                            "diversification": self.diversification,
                            "variance": self.higher_moment,
                            "volatility": self.volatility,
                            "skew": self.higher_moment,
                            "kurt": self.higher_moment,
                            "moment": self.higher_moment,
                            "risk_parity": self.risk_parity,
                            "expected_return": self.expected_return,
                            "sharpe": self.sharpe,
                            "beta": self.beta,
                            "treynor": self.treynor,
                            "jenson_alpha": self.jenson_alpha}

    # Weight Related
    def leverage(self, w):
        """
        Calculates the leverage based on weight
        * Short positions contribute to leverage
        :param w: np.ndarray, weight vector
        :return: float
        """
        return np.sum(np.sqrt(np.square(w)))

    def num_assets(self, w):
        """
        Calculates the number of assets based on weight
        :param w: np.ndarray, weight vector
        :return: int
        """
        return len(w[np.round(w, 3) != 0])

    def concentration(self, w, top_holdings):
        """
        Calculates the % concentration of top holdings
        :param w: np.ndarray, weight vector
        :param top_holdings: number of top holdings by absolute value of weight (leverage)
        :return: float
        """
        return -np.sum(np.partition(-np.sqrt(np.square(w)), top_holdings)[:top_holdings])/np.sum(np.sqrt(np.square(w)))


    # Risk-only Metric
    def correlation(self, w):

        """
        Calculates the correlation factor of the portfolio
        Must pass in a covariance matrix
        :param w: np.ndarray, weight vector
        :return: float
        """
        if self.moment != 2:
            raise DimException("Did not pass in a covariance matrix")
        # Assume Covariance Matrix is passed in
        corr_mat = self.moment_mat * np.dot(((np.diag(self.moment_mat)) ** -0.5).reshape(-1, 1),
                                            ((np.diag(self.moment_mat)) ** -0.5).reshape(1, -1))
        return w @ corr_mat @ w.T

    def diversification(self, w):
        """
        Calculates the diversifcation factor of the portfolio
        Must pass in a covariance matrix
        :param w: np.ndarray, weight vector
        :return: float
        """
        if self.moment != 2:
            raise DimException("Did not pass in a covariance matrix")
        std_arr = np.diag(self.moment_mat) ** 0.5
        return (w @ std_arr)/np.sqrt(w @ self.moment_mat @ w.T)

    def volatility(self, w):
        """
        Calculates the volatility of the portfolio
        Must pass in a covariance matrix
        :param w: np.ndarray, weight vector
        :return: float
        """
        if self.moment != 2:
            raise DimException("Did not pass in a covariance matrix")
        return np.sqrt(self.higher_moment(w))

    def higher_moment(self, w):
        """
        Calculates the moment of the portfolio
        Moment=2 -> Variance, Moment=3 -> Skewness, Moment=4 -> Kurtosis
        :param w: np.ndarray, weight vector
        :return: float
        """
        temp = w
        for iteration in range(self.moment - 2):
            temp = np.kron(w, temp)
        return w @ self.moment_mat @ temp.T

    def risk_parity(self, w):
        """
        Calculates the risk parity of the portfolio
        Must pass in a covariance matrix
        :param w: np.ndarray, weight vector
        :return: float
        """
        if self.moment != 2:
            raise DimException("Did not pass in a covariance matrix")
        return 0.5 * w @ self.moment_mat @ w.T - np.sum(np.log(w))/len(self.assets)
        # return 0.5 * cp.quad_form(self.weight_param, self.moment_mat) - cp.sum(cp.log(self.weight_param))/len(self.assets)

    # Risk-Reward
    def expected_return(self, w):
        """
        Calculates the expected return of the portfolio
        :param w: np.ndarray, weight vector
        :return: float
        """
        return w @ self.ret_vec

    def sharpe(self, w, risk_free):
        """
        Calculates the sharpe ratio of the portfolio. Note that if the covariance matrix is semivariance matrix, then it calculates the sortino ratio
        Must pass in a covariance matrix
        :param w: np.ndarray, weight vector
        :param risk_free: float, risk free rate of return, must be greater than 0 to satisfy CAPM assumptions
        :return: float
        """
        if self.moment != 2:
            raise DimException("Did not pass in a covariance matrix")
        return (self.expected_return(w) - risk_free)/self.volatility(w)

    def beta(self, w):
        """
        Calculates the beta of the portfolio.
        :param w: np.ndarray, weight vector
        :return: beta
        """
        return w @ self.beta_vec

    def treynor(self, w, risk_free):
        """
        Calculates the treynor ratio of the portfolio. Beta vector is required for the calculation.

        :param w: np.ndarray, weight vector
        :param risk_free: float, risk free rate of return
        :return: float
        """
        return (self.expected_return(w) - risk_free)/self.beta(w)

    def jenson_alpha(self, w, risk_free, market_return):
        """
        Calculates the Jenson's alpha of the portfolio. Beta vector is required for the calculation.
        :param w: np.ndarray, weight vector
        :param risk_free: float, risk free rate of return
        :param market_return: float, expected market return
        :return: float
        """
        return self.expected_return(w) - risk_free - self.beta(w) * (market_return - risk_free)

    # Numeric
    def inverse_volatility(self, leverage):
        """
        Calculates weight based on the inverse volatility of individual portfolios
        :return: np.ndarray
        """

        if self.moment != 2:
            raise DimException("Did not pass in a covariance matrix")

        std_arr = np.diag(self.moment_mat) ** 0.5
        return (1/std_arr)/np.sum(1/std_arr) * leverage

    def inverse_variance(self, leverage):
        """
         Calculates weight based on the inverse variance of individual portfolios
         :return: np.ndarray
         """
        if self.moment != 2:
            raise DimException("Did not pass in a covariance matrix")

        var_arr = np.diag(self.moment_mat)
        return (1/var_arr)/np.sum(1/var_arr) * leverage

    def equal_weight(self, leverage):
        """
        Constructs an equal weight portfolio (Long only)
         :return: np.ndarray
        """
        return np.repeat(leverage/len(self.assets), len(self.assets))

    def market_cap_weight(self, leverage):
        """
         Construct a portfolio based on market cap weight (Long only)
         :return: np.ndarray
         """
        market_cap_info = MetricGenerator.market_cap_data(self.assets)
        return market_cap_info/np.sum(market_cap_info) * leverage

    @staticmethod
    def market_cap_data(assets):
        """
        Helper function to get market cap information from yahoo finance
        :param assets: List[str], ticker symbols
        :return: np.ndarray
        """
        return data.get_quote_yahoo(assets)['marketCap'].values




import numpy as np
from .Metrics import MetricGenerator


class ObjectiveGenerator(MetricGenerator):

    def __init__(self, ret_vec, moment_mat, moment, assets, beta_vec):

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
        if objective_type in ["equal_weight", "market_cap_weight", "inverse_volatility", "inverse_variance"]:
            return self.method_dict[objective_type](**kwargs)
        return self.method_dict[objective_type]

    # Classic Equation
    def efficient_frontier(self, w, aversion):
        return -(self.expected_return(w) - aversion * self.higher_moment(w))

    # Risk Related
    def equal_risk_parity(self, w):
        return self.risk_parity(w)

    def min_correlation(self, w):
        return self.correlation(w)

    def min_volatility(self, w):
        return self.volatility(w)

    # def min_variance(self, w):
    #     return self.variance(w)
    #
    # def min_skew(self, w):
    #     return self.min_moment(w)
    #
    # def min_kurt(self, w):
    #     return self.min_moment(w)

    def min_moment(self, w):
        return self.higher_moment(w)

    def max_diversification(self, w):
        return -self.diversification(w)

    # Return related
    def max_return(self, w):
        return -self.expected_return(w)

    # Metrics related
    def max_sharpe(self, w, risk_free):
        return -self.sharpe(w, risk_free)

    # Make beta close to zero
    def min_beta(self, w):
        return np.sqrt(np.square(self.beta(w)))

    def max_treynor(self, w, risk_free):
        return -self.treynor(w, risk_free)

    def max_jenson_alpha(self, w, risk_free, market_return):
        return -self.jenson_alpha(w, risk_free, market_return)



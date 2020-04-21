import warnings

from .Metrics import *

### https://www.portfolioprobe.com/features/constraints/


class ConstraintGenerator(MetricGenerator):


    def __init__(self, ret_vec, moment_mat, moment, assets, beta_vec):

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
                            "variance": self.variance_const,
                            "skew": self.skew_const,
                            "kurt": self.kurt_const,
                            "moment": self.moment_const}

    def create_constraint(self, constraint_type, **kwargs):
        return self.method_dict[constraint_type](**kwargs)

    # Weight Only
    def weight(self, weight_bound, leverage): # Checked
        init_bound = (0,1)
        individual_bound = ConstraintGenerator.construct_weight_bound(self.ret_vec.shape[0], init_bound, weight_bound)

        # total_bound = [{'type': 'eq', 'fun': lambda w:  np.sum(w) - total_weight}]
        total_leverage = [{'type': 'eq', 'fun': lambda w: -self.leverage(w) + leverage}]
        return individual_bound, total_leverage

    # def leverage(self, leverage): # Checked
    #     return [{'type': 'ineq', 'fun': lambda w: -np.sum(np.sqrt(np.square(w))) + leverage}]

    def num_assets_const(self, num_assets): # Checked
        if self.ret_vec.shape[0] <= num_assets:
            warnings.warn("""The number of assets to hold exceeds the number of assets available, 
            default to a 1 asset only scenario""")
            num_assets = self.ret_vec.shape[0] - 1
        non_holdings = self.ret_vec.shape[0] - num_assets
        # return [{'type': 'eq', 'fun': lambda w: self.num_assets(w) - num_assets}]
        return [{'type': 'eq', 'fun': lambda w: np.sum(np.partition(np.sqrt(np.square(w)), non_holdings)[:non_holdings])}]
        # return [cp.sum_smallest(cp.abs(self.weight_param), self.ret_vec.shape[0] - num_assets) <= 0.0001]

    def concentration_const(self, top_holdings, top_concentration): # Checked
        if self.ret_vec.shape[0] <= top_holdings:
            warnings.warn("""Number of Top Holdings Exceeds Total Available Assets. 
            Will default top_holdings to be number of holdings available""")
            top_holdings = self.ret_vec.shape[0]
        # return [{"type": "ineq", "fun": lambda w: -self.concentration(w, top_holdings) + top_concentration}]
        # return [cp.sum_largest(cp.norm(self.weight_param, 1), top_holdings) <= top_concentration]
        return [{"type": "ineq", "fun": lambda w: np.sum(
            np.partition(-np.sqrt(np.square(w)), top_holdings)[:top_holdings]) / np.sum(
            np.sqrt(np.square(w))) + top_concentration}]


    ### Market Data Needed/Calculation Needed
    def market_neutral_const(self, bound):

        market_cap_weight = self.market_cap_data()
        return [{"type": "ineq", "fun": lambda w: market_cap_weight @ self.moment_mat @ w - bound[0]},
                {"type": "ineq", "fun": lambda w: -(market_cap_weight @ self.moment_mat @ w - bound[1])}]


    # Return related
    def expected_return_const(self, bound): # Checked
        bound = ConstraintGenerator.construct_const_bound(bound, True, 10)
        return [{"type": "ineq", "fun": lambda w: self.expected_return(w) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.expected_return(w) + bound[1]}]

    def sharpe_const(self, risk_free, bound):
        bound = ConstraintGenerator.construct_const_bound(bound, True, 10)

        return [{"type": "ineq", "fun": lambda w: self.sharpe(w, risk_free) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.sharpe(w, risk_free) + bound[1]}]

    def beta_const(self, bound):
        bound = ConstraintGenerator.construct_const_bound(bound, False, 1)
        return [{"type": "ineq", "fun": lambda w: self.beta(w) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.beta(w) + bound[1]}]

    def treynor_const(self, bound, risk_free):
        bound = ConstraintGenerator.construct_const_bound(bound, True, 10)
        return [{"type": "ineq", "fun": lambda w: self.treynor(w, risk_free) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.treynor(w, risk_free)  + bound[1]}]

    def jenson_alpha_const(self, bound, risk_free, market_return):
        bound = ConstraintGenerator.construct_const_bound(bound, True, 10)
        return [{"type": "ineq", "fun": lambda w: self.jenson_alpha(w, risk_free, market_return) - bound[0]},
                {"type": "ineq", "fun": lambda w: -self.jenson_alpha(w, risk_free, market_return) + bound[1]}]

    # Risk related constraints
    def volatility_const(self, bound):
        bound = ConstraintGenerator.construct_const_bound(bound, False, 0)
        return [{"type": "ineq", "fun": lambda w: self.volatility(w)  + bound[0]},
                {"type": "ineq", "fun": lambda w: -self.volatility(w) + bound[1]}]

    def variance_const(self, bound):
        if self.moment != 2:
            raise DimException("Did not pass in a correlation/covariance matrix")
        bound = ConstraintGenerator.construct_const_bound(bound, False, 0)
        return [{"type": "ineq", "fun": lambda w: self.variance(w) + bound[0]},
                {"type": "ineq", "fun": lambda w: -self.variance(w) + bound[1]}]

    def skew_const(self, bound):
        if self.moment != 3:
            raise DimException("Did not pass in a coskewness matrix")
        bound = ConstraintGenerator.construct_const_bound(bound, False, 0)
        return [{"type": "ineq", "fun": lambda w: self.higher_moment(w) + bound[0]},
                {"type": "ineq", "fun": lambda w: -self.higher_moment(w) + bound[1]}]

    def kurt_const(self, bound):
        if self.moment != 4:
            raise DimException("Did not pass in a cokurtosis matrix")

        bound = ConstraintGenerator.construct_const_bound(bound, False, 0)
        return [{"type": "ineq", "fun": lambda w: self.higher_moment(w) + bound[0]},
                {"type": "ineq", "fun": lambda w: -self.higher_moment(w) + bound[1]}]

    def moment_const(self, bound):
        bound = ConstraintGenerator.construct_const_bound(bound, False, 0)
        return [{"type": "ineq", "fun": lambda w: self.higher_moment(w) + bound[0]},
                {"type": "ineq", "fun": lambda w: -self.higher_moment(w) + bound[1]}]
    
    @staticmethod
    def construct_const_bound(bound, minimum, opposite_value):
        if isinstance(bound, (int, float)):
            warnings.warn(f"""Only one bound is given, will set the {'maximum' if minimum else 'minimum'} value to be {opposite_value}""")
            if minimum:
                bound = (bound, opposite_value)
            else:
                bound = (opposite_value, bound)
        return bound
    
    @staticmethod
    def construct_weight_bound(size, init_bound, weight_bound):

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



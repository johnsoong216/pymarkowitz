"""
BackTester class holds a sample configuration and a weighting strategy


"""
import bt
from .Returns import ReturnGenerator
from .Moments import MomentGenerator
from .Optimizer import Optimizer


class Config:
    lookback = 30
    calc_return_dict = {"method": "daily"}
    calc_mean_return_dict = {"method": "arithmetic"}
    calc_moment_dict = {}
    calc_beta_dict = {}
    beta = "spy"
    objective = {"objective_type": "min_volatility"}
    constraints = [{"constraint_type": "weight", "weight_bound": (0, 0.25), "leverage": 1}]


class WeighMarkowitz(bt.Algo):

    def __init__(self, Config):
        super(WeighMarkowitz, self).__init__()
        self.lookback = Config.lookback
        self.beta = Config.beta
        self.config = Config

    def __call__(self, target):
        selected = target.temp['selected']
        non_benchmark = [asset for asset in selected if asset != self.beta]

        t0 = target.universe[selected].index.get_loc(target.now)
        if t0 >= self.lookback:
            price_mat = target.universe[selected].iloc[t0 - self.lookback:t0].dropna(how='any')
            ret_generator = ReturnGenerator(price_mat)

            daily_return = ret_generator.calc_return(**self.config.calc_return_dict)
            mean_return = ret_generator.calc_mean_return(**self.config.calc_mean_return_dict)[non_benchmark]
            mom_generator = MomentGenerator(daily_return[non_benchmark])
            mom_matrix = mom_generator.calc_cov_mat(**self.config.calc_moment_dict)
            beta_vec = mom_generator.calc_beta(beta_vec=daily_return[self.beta], **self.config.calc_beta_dict)

            PortOpt = Optimizer(mean_return, mom_matrix, beta_vec)
            PortOpt.add_objective(self.config.objective["objective_type"],
                                  **{objective_arg: self.config.objective[objective_arg] for objective_arg in
                                     self.config.objective if objective_arg not in ["objective_type"]})
            for c in self.config.constraints:
                PortOpt.add_constraint(c["constraint_type"],
                                       **{c_arg: c[c_arg] for c_arg in c if c_arg not in ["constraint_type"]})
            PortOpt.solve()
            target.temp['weights'] = PortOpt.summary()[0]
        return True


# Pymarkowitz

<p align="left">
    <a href="https://www.python.org/">
        <img src="https://ForTheBadge.com/images/badges/made-with-python.svg"
            alt="python"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square"
            alt="MIT license"></a> &nbsp;
</p>

**Pymarkowitz** is an open source library for implementing portfolio optimisation. This library extends beyond the classical mean-variance optimization and takes into account a variety of risk and reward metrics, as well as the skew/kurtosis of assets.

**Pymarkowitz** can aid your decision-making in portfolio allocation in a risk-efficient manner. Pymarkowitz covers major objectives and constraints related with major types of risk and reward metrics, as well as simulation to examine the relationship between all these metrics. The flexibility in its implementation gives you the maximum discretion to customize and suit it to your own needs. 


*Disclaimer: This library is for educational and entertainment purpose only. Please invest with due diligence at your own risk.

Head over to the directory **demos** to get an in-depth look at the project and its functionalities, or continue below to check out some brief examples.

---

## Table of Contents


- [Installation](#installation)
- [Features](#features)
- [Get In Touch](#get-in-touch)
- [Reference](#reference)
- [License](#license)

---

## Installation

### Setup

> install directly using pip

```shell
$ pip install pymarkowitz
```

> install from github

```shell
$ pip install git+https://github.com/johnsoong216/pymarkowitz.git
```

### Development

> For development purposes you can clone or fork the repo and hack right away!

```shell
$ git clone https://github.com/johnsoong216/pymarkowitz.git
```
---

## Features
- [Preprocessing](##preprocessing)
- [Optimization](##optimization)
- [Simulation](##simulation)
- [Backtesting](##backtesting)


---
### Preprocessing

> First step is to import all availble modules

```python
import numpy as np
import pandas as pd
from pymarkowitz import *

```
> Read data with pandas. The dataset is available in the **datasets** directory. I will select 15 random stocks with 1000 observations

```python

sp500 = pd.read_csv("datasets/sp500_1990_2000.csv", index_col='DATE').drop(["Unnamed: 0"], axis=1)
selected = sp500.iloc[:1000, np.random.choice(np.arange(0, sp500.shape[1]), 15, replace=False)]

```
> Use a ReturnGenerator to compute historical mean return and daily return. Note that there are a variety of options to compute rolling/continuous/discrete returns. Please refer to the **Return.ipynb** jupyter notebook in **demo** directory

```python

ret_generator = ReturnGenerator(selected)
mu_return = ret_generator.calc_mean_return(method='geometric')
daily_return = ret_generator.calc_return(method='daily')

```
> Use a MomentGenerator to compute covariance/coskewness/cokurtosis matrix and beta. Note that there are a variety of options to compute the comoment matrix and asset beta, such as with semivariance, exponential and customized weighting. Normalizing matrices are also supported. Please refer to the **Moment(Covariance).ipynb** jupyter notebook in **demo** directory

```python

benchmark = sp500.iloc[:1000].pct_change().dropna(how='any').sum(axis=1)/sp500.shape[1]
cov_matrix = mom_generator.calc_cov_mat()
beta_vec = mom_generator.calc_beta(benchmark)

```

> Construct higher moment matrices by calling

```python


coskew_matrix = mom_generator.calc_coskew_mat()
cokurt_matrix = mom_generator.calc_cokurt_mat()
coseventh_matrix = mom_generator.calc_comoment_mat(7)

```

> Construct an Optimizer

```python

PortOpt = Optimizer(mu_return, cov_matrix, beta_vec)

```

### Optimization

> Please refer to the **Optimization.ipynb** jupyter notebook in **demo** directory for more detailed explanations.


> Set your Objective. 

```python

### Call PortOpt.objective_options() to look at all available objectives

PortOpt.add_objective("min_volatility")

```

> Set your Constraints. 

```python

### Call PortOpt.constraint_options() to look at all available constraints.

PortOpt.add_constraint("weight", weight_bound=(-1,1), leverage=1) # Portfolio Long/Short
PortOpt.add_constraint("concentration", top_holdings=2, top_concentration=0.5) # Portfolio Concentration

```

> Solve and Check Summary


```python
PortOpt.solve()
weight_dict, metric_dict = PortOpt.summary(risk_free=0.015, market_return=0.07, top_holdings=2)


# Metric Dict Sample Output
{'Expected Return': 0.085,
 'Leverage': 1.0001,
 'Number of Holdings': 5,
 'Top 2 Holdings Concentrations': 0.5779,
 'Volatility': 0.1253,
 'Portfolio Beta': 0.7574,
 'Sharpe Ratio': 0.5586,
 'Treynor Ratio': 0.0924,
 "Jenson's Alpha": 0.0283}
 
# Weight Dict Sample Output
{'GIS': 0.309, 'CINF': 0.0505, 'USB': 0.104, 'HES': 0.2676, 'AEP': 0.269}

```

### Simulation

> Simulate and Select the Return Format (Seaborn, Plotly, DataFrame). DataFrame Option will also have the random weights used in each iteration.

> Please refer to the **Simulation.ipynb** jupyter notebook in **demo** directory for more detailed explanations.


```python

### Call Portopt.metric_options to see all available options for x, y axis

PortOpt.simulate(x='expected_return', y='sharpe', y_var={"risk_free": 0.02}, iters=10000, weight_bound=(-1, 1), leverage=1, ret_format='sns')

```
![Sharpe VS Return](https://github.com/johnsoong216/pymarkowitz/blob/master/images/return_vs_sharpe.png)


### Backtesting

> To be updated

---


## Get In Touch

- Please reach out to me through email at johnsoong216@hotmail.com. Love to get connected and Chat!

---

## Reference

Calculations of **Correlation, Diversifcation & Risk Parity Factors**:
<br>
https://investresolve.com/file/pdf/Portfolio-Optimization-Whitepaper.pdf

Calculations for **Sharpe, Sortino, Beta, Treynor, Jenson's Alpha**:
<br>
https://www.cfainstitute.org/-/media/documents/support/programs/investment-foundations/19-performance-evaluation.ashx?la=en&hash=F7FF3085AAFADE241B73403142AAE0BB1250B311
<br>
https://www.investopedia.com/terms/j/jensensmeasure.asp
<br>
https://www.investopedia.com/ask/answers/070615/what-formula-calculating-beta.asp
<br>

Calculations for **Higher Moment Matrices**:
<br>
https://cran.r-project.org/web/packages/PerformanceAnalytics/vignettes/EstimationComoments.pdf
<br>


---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2020 ©

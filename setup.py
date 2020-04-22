from setuptools import setup

setup(name='pymarkowitz',
      version='0.1.0',
      description='pymarkowitz',
      url='https://github.com/johnsoong216/pymarkowitz',
      author='johnsoong216',
      author_email='johnsoong216@hotmail.com',
      license='MIT',
      install_requires=["numpy>=1.17.1", "pandas>=0.25.3", "pandas-datareader>=0.8.1", "sklearn>=0.0", "seaborn>=0.9.0", "plotly>=4.1.1", "matplotlib>=3.2.1"],
      zip_safe=False)
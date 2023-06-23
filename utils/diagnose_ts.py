import pandas as pd
import numpy as np

from typing import List, Tuple, Optional, Union
from scipy.stats import shapiro
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import plotly_express as px


class TimeSeriesAnalysis:
    """
    Class for time series analysis. As an input it takes a any array-like object with a datetime index
    or a pd.DataFrame with a datetime index or with a column with datetime values, and a column with target values.
    """
    def __init__(
        self, ts,
        data_time_column: Optional[str] = None,
        target_column: Optional[str] = None,
        is_residuals: bool = False
        ) -> None:
        self._set_ts(ts, data_time_column, target_column)
        self.is_residuals = is_residuals
    
    def _set_ts(self, ts, data_time_column: Optional[str] = None, target_column: Optional[str] = None) -> None:
        if isinstance(ts, pd.Series):
            self.data_time_column = ts.index.name
            self.target_column = ts.name
            self.ts = ts
            self.ts.index = pd.to_datetime(self.ts.index)
        elif isinstance(ts, pd.DataFrame):
            self.ts = ts.copy()
            
            if data_time_column is None:
                self.data_time_column = ts.index.name
                self.ts.index = pd.to_datetime(self.ts.index)
            else:
                assert data_time_column in ts.columns, f"data_time_column {data_time_column} not in ts.columns"
                self.data_time_column = data_time_column
                self.ts[data_time_column] = pd.to_datetime(self.ts[data_time_column])
                self.ts = self.ts.set_index(data_time_column)
                        
            if target_column is None:
                self.target_column = ts.columns[0]
            else:
                assert target_column in ts.columns, f"target_column {target_column} not in ts.columns"
                self.target_column = target_column
            self.ts = self.ts[self.target_column]
        else:
            raise ValueError(f"ts should be a pd.Series or pd.DataFrame with a datetime index, got {type(ts)}")



    def mean_is_zero(self,atol: float = 0.01, verbose: bool = True):
        mean = np.mean(self.ts)
        if verbose:
            print(f"Mean: {mean}")
        return np.isclose(mean, 0, atol = atol)

    def is_uncorrelated(self, lag=1):
        autocorr = acf(self.ts.to_numpy(), nlags=lag)
        return np.all(np.isclose(autocorr, 0))

    def is_homoscedastic(self):
        # Perform the Augmented Dickey-Fuller test
        result = adfuller(self.ts)
        # If p-value is small, we can reject the null hypothesis of a unit root, and the series is stationary
        return result[1] < 0.05

    def is_normal(self):
        # perform the Shapiro-Wilk test
        result = shapiro(self.ts)
        return result[1] > 0.05
    
    
    def distribution_plot(self, n_bins: int = 20):
        fig = px.histogram(self.ts, x=self.target_column, nbins=n_bins)
        return fig            
    
    def residuals_test(self, verbose: bool = True):
        _mean_is_zero = self.mean_is_zero(verbose=verbose)
        print(f"Mean is zero: {_mean_is_zero}")
        print(f"Is uncorrelated: {self.is_uncorrelated()}")
        print(f"Is homoscedastic: {self.is_homoscedastic()}")
        print(f"Is normal: {self.is_normal()}")
        if verbose:
            self.distribution_plot().show()
        
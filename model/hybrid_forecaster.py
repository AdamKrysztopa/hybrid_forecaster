import pandas as pd
from statsforecast import StatsForecast
# from statsfrecast.model import AutoARIMA
# from sklearn.ensemble import RandomForestRegressor


class HybridForecaster:
    def __init__(self,
                statfrecast_model: object,
                ml_model: object,
                feature_selector: object = None,
                forecast_horizon: int = 12,
                season_length: int = 12,
                freq: str = 'MS',
                lag_multiplier: int = 2):
        self.statfrecast_model = statfrecast_model
        self.ml_model = ml_model()
        self.feature_selector = feature_selector
        self.forecast_horizon = forecast_horizon
        self.season_length = season_length
        self.freq = freq
        self.lag_multiplier = lag_multiplier
        self.stf_model = StatsForecast(
            models = [self.statfrecast_model(season_length=self.season_length)],
            freq = self.freq,
            n_jobs = -1,
            verbose = False
        )
        
        self._is_fitted = False
        self.statsforecast_residuals = None
        self.final_residuals = None
        self.in_sample_preds = None
        self.X_lagged_inference = None
        self.unique_id = None
    
    #method fit takes ts (timeseries) and X (exogenous variables) as input)
    # fit has 4 steps:
    # 1. fit the statsforecast model to the timeseries ts (_fit_statistical_model)
    # 2. prepare set of the laged exogenous variables (_prepare_lagged_exog) min_lag = forecast_horizon, max_lag = 2 * forecast_horizon
    # 3. select features using the feature selector (_select_features), if feature selector is None, then use all features
    # 4. fit the ml model to the residuals of the statsforecast model (_fit_ml_model)
    
    def _fit_statistical_model(self, ts: pd.DataFrame):
        self.stf_model.fit(ts)
        return pd.Series(self.stf_model.fitted_[0][0].predict_in_sample()['fitted'], index=ts['ds'])
    
    @staticmethod
    def _lag_df(X, lag, lag_period, rename_cols: bool = True):\
        #lagged_period = 'M' # 'M' for month, 'W' for week, 'D' for day, 'Q' for quarter, 'Y' for year
        assert lag_period in {'M', 'MS', 'W', 'D', 'Q', 'Y'}, "legged_period should be one of the following:"\
            " 'M', 'MS' 'W', 'D', 'Q', 'Y'"
        laged_X = X.copy()
        if lag_period == 'D':
            laged_X.index = laged_X.index + pd.DateOffset(days=lag)
        elif lag_period == 'W':
            laged_X.index = laged_X.index + pd.DateOffset(weeks=lag)
        elif lag_period in {'M', 'MS'}:
            laged_X.index = laged_X.index + pd.DateOffset(months=lag)
        elif lag_period == 'Q':
            laged_X.index = laged_X.index + pd.DateOffset(month=3*lag)
        else:
            laged_X.index = laged_X.index + pd.DateOffset(years=lag)
        if rename_cols:
            laged_X.columns = [f"{col}_lag_{lag}" for col in laged_X.columns]
        return laged_X
    
    def _prepare_lagged_exog(
        self,
        X: pd.DataFrame,
        number_of_lags: int = 12,
        lag_multiplier: int = 2,
        lag_period: str = 'M'
        ):
        
        X_lagged = pd.DataFrame(index=X.index)
        
        assert lag_multiplier > 1, "lag_multiple should be greater or equal to 2"
        assert number_of_lags >= 1, "number_of_lags should be greater or equal to 1"
        assert number_of_lags <= (lag_multiplier-1) * self.forecast_horizon, "number_of_lags should be less or equal "\
            "to (lag_multiple-1) * forecast_horizon"
        
        for lag in range(self.forecast_horizon, 
                         lag_multiplier * self.forecast_horizon, 
                         ((lag_multiplier - 1) * self.forecast_horizon)//number_of_lags)[:number_of_lags]: 
            temp_X = self._lag_df(X, lag, lag_period)
            X_lagged = pd.merge(X_lagged, temp_X, left_index=True, right_index=True, how='outer')
        X_lagged_train = X_lagged.loc[X_lagged.index <= X.index.max()]
        X_lagged_inference = X_lagged.loc[X_lagged.index > X.index.max()][:self.forecast_horizon]
                
        return X_lagged_train, X_lagged_inference
    
    def _fit_ml_model(self, X: pd.DataFrame, y: pd.Series):
        
        self.ml_model.fit(X = X, y = y)
        
        return pd.Series(self.ml_model.predict(X = X), index=X.index)
    
    def fit(self, ts: pd.DataFrame, X: pd.DataFrame = None):
        
        self.unique_id = ts['unique_id'].unique()[0]

        # 1. fit the statsforecast model to the timeseries ts
        X = X.copy()
        if 'ds' in X.columns:
            X['ds'] = pd.to_datetime(X['ds'])
            X = X.set_index('ds')
            
        # 2. prepare set of the laged exogenous variables
        # 2.1. prepare laged exogenous variables
        # 2.2. prepare laged exogenous variables for inference mode, i.e. shifted by forecast_horizon to be used to predict the future
        
        # assumption thant no more than 12 lags are used
        X_lagged, self.X_lagged_inference = self._prepare_lagged_exog(
            X,
            number_of_lags=min(self.forecast_horizon,12),
            lag_multiplier=self.lag_multiplier,
            lag_period=self.freq)

        #it is done, becasuse some nans are in laged
        #TODO: check why it is so

        self.X_lagged_inference = self.X_lagged_inference.dropna(axis=1)
        
        X_lagged = X_lagged.dropna()[self.X_lagged_inference.columns]

        # cuts ts the history depth to the same as X_lagged
        ts = ts.loc[ts['ds'] >= X_lagged.index.min()]
        stat_in_sample_preds = self._fit_statistical_model(ts)
        #since now, all data is in the same format, i.e. ds is index
        ts = ts.set_index('ds')
        self.statsforecast_residuals = ts['y'] - stat_in_sample_preds  
        
        # fit non-linear model to the residuals     
        ml_in_sample_preds = self._fit_ml_model(X_lagged, self.statsforecast_residuals)
        
        self.in_sample_preds = stat_in_sample_preds + ml_in_sample_preds
        self.final_residuals = ts['y'] - self.in_sample_preds
        self._is_fitted = True
        
        return self
    
    def predict(self):
            
        assert self._is_fitted, "Model is not fitted yet"
        
        ml_inference_preds = self.ml_model.predict(self.X_lagged_inference)
        ml_inference_preds = pd.Series(ml_inference_preds, index=self.X_lagged_inference.index)
        
        stat_inference_preds = self.stf_model.predict(self.forecast_horizon).set_index('ds').squeeze()
        
        inference_preds = stat_inference_preds + ml_inference_preds
        
        result = pd.DataFrame(inference_preds).reset_index()
        result.columns = ['ds', 'y_hat']
        result['unique_id'] = self.unique_id
        
        return result[['unique_id', 'ds', 'y_hat']]
        
    def get_predict_in_sample(self):
        
        assert self._is_fitted, "Model is not fitted yet"
        
        result = pd.DataFrame(self.in_sample_preds).reset_index()
        result.columns = ['ds', 'y_hat']
        result['unique_id'] = self.unique_id
        
        return result[['unique_id', 'ds', 'y_hat']]
    
    def get_residuals(self):
        
        assert self._is_fitted, "Model is not fitted yet"
        
        statsforecast_residuals = pd.DataFrame(self.statsforecast_residuals).reset_index()
        statsforecast_residuals.columns = ['ds', 'sf_residuals']
        
        result = pd.DataFrame(self.final_residuals).reset_index()
        result.columns = ['ds', 'residuals']
        
        result = pd.merge(result, statsforecast_residuals, on='ds', how='left')
        result['unique_id'] = self.unique_id
        
        return result[['unique_id', 'ds','sf_residuals' ,'residuals']]
        
        
    
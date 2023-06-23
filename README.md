# HybridForecaster

The `HybridForecaster` is a Python class that combines statistical forecasting models with machine learning models to predict time series data. The class is designed to be flexible, allowing users to specify their own statistical and machine learning models, as well as a feature selector.

## Features

The `HybridForecaster` class includes the following features:

- **Statistical Forecasting Model**: The class uses a statistical forecasting model to make initial predictions on the time series data. This model is specified by the user when the `HybridForecaster` is initialized.

- **Machine Learning Model**: The class uses a machine learning model to predict the residuals (the difference between the actual values and the predictions made by the statistical model). This model is also specified by the user when the `HybridForecaster` is initialized.

- **NOT IMPLEMENTED!** **Feature Selector**: The class allows the user to specify a feature selector, which is used to select the most relevant features from the lagged exogenous variables.

- **Lagged Exogenous Variables**: The class prepares a set of lagged exogenous variables, which are used as input to the machine learning model.

- **In-Sample Predictions**: The class provides a method to get the in-sample predictions, which are the predictions made on the training data.

- **Residuals**: The class provides a method to get the residuals, which are the difference between the actual values and the predictions made by the model.

## Usage

To use the `HybridForecaster` class, you need to import it and initialize it with your chosen statistical forecasting model, machine learning model, and feature selector. You can then use the `fit` method to train the model on your time series data and the `predict` method to make predictions.

## To-Do List

- **Implement Featue Selection**:The feature selector should be designed to operate on a pandas DataFrame. It should use the DataFrame's indices and utilize column names for feature identification and selection.

- **Improve Handling of Missing Values**: Currently, the class drops any columns with missing values in the lagged exogenous variables. This could potentially result in loss of important information. A better approach would be to fill the missing values using a suitable method.

- **Add More Flexibility to Lagged Exogenous Variables**: Currently, the class only supports lagging the exogenous variables by a fixed number of periods. It would be useful to allow the user to specify different lag periods for different variables.

- **Add forevard method**: Currently predict method ueses up fron defined forecast horizon and basing on that, the lagged exogenous data are prepared for inference mode. It is usefull that X lagged for inference mode could be externally added.

- **Add Model Evaluation Metrics**: It would be useful to add methods to calculate common model evaluation metrics, such as mean absolute error (MAE), root mean square error (RMSE), and mean absolute percentage error (MAPE).

- **Add Model Diagnostics**: It would be useful to add methods to perform model diagnostics, such as checking the residuals for autocorrelation and plotting the actual values against the predicted values.

## Example Notebook

An example Jupyter notebook, `Examples.ipynb`, is provided with the `HybridForecaster` class. This notebook demonstrates how to use the class to analyze demand data.

The data used in the example is a time series of demand for a particular item from a store. The data includes the following columns:

- `id`: A unique identifier for each record.
- `date`: The date of the record.
- `store`: The identifier of the store.
- `item`: The identifier of the item.

[Store Item Demand Forecasting Challenge Data Source](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)

### To-Do List

- improve plots quality

- add some explanations around the study

- verify different models (statistical and ML ones)

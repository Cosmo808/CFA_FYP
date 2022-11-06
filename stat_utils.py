import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


class Stat_utils:
    @staticmethod
    def linear_regression_params(X, y):
        lr_model = LinearRegression().fit(X, y)
        params = np.append(lr_model.intercept_, lr_model.coef_)
        predictions = lr_model.predict(X)

        newX = np.append(np.ones((len(X), 1)), X, axis=1)
        MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX[0]))

        var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params / sd_b

        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]

        sd_b = np.round(sd_b, 3)
        ts_b = np.round(ts_b, 3)
        p_values = np.round(p_values, 3)
        params = np.round(params, 5)

        params_display = pd.DataFrame()
        params_display["Coefficients"] = params
        params_display["Standard Errors"] = sd_b
        params_display["t-values"] = ts_b
        params_display["p-values"] = p_values
        return params_display


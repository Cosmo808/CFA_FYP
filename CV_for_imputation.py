import os
import numpy as np
import pandas as pd
from pd_data_preprocess import Pandas_data
from stat_utils import Stat_utils
from sklearn.metrics import mean_squared_error as mse
from impyute.imputation.cs import mice
import statsmodels.formula.api as smf
from statsmodels.api import load


if __name__ == "__main__":
    data = Pandas_data()
    stat = Stat_utils()
    pd_age = data.age.iloc[:, 2:4]
    pd_gmv = data.gmv
    pd_sex = data.sex
    pd_eth = data.eth

    pd_age_gmv_sex_eth = pd.concat([pd_age, pd_gmv, pd_sex, pd_eth], axis=1)  # 502411
    pd_age_gmv_sex_eth = pd_age_gmv_sex_eth.dropna(subset=['gmv_2'])  # 42806
    pd_age_gmv_sex_eth = pd_age_gmv_sex_eth.dropna(subset=['sex'])  # 41766
    pd_age_gmv_sex_eth = pd_age_gmv_sex_eth.dropna(subset=['eth_0'])  # 41758
    pd_index = pd_age_gmv_sex_eth.index

    pd_filter_gmv_3 = pd_age_gmv_sex_eth.dropna(subset=['gmv_3'])
    pd_index_gmv_3 = pd_filter_gmv_3.index
    cv_length = int(len(pd_index_gmv_3) / 10)

    gmv_rmse_0 = []
    age_rmse_0 = []
    gmv_rmse_1 = []
    age_rmse_1 = []
    for i in range(10):
        if i == 9:
            cv_index = pd_index_gmv_3[(i * cv_length):]
        else:
            cv_index = pd_index_gmv_3[(i * cv_length):((i + 1) * cv_length)]

        gmv_3 = pd_filter_gmv_3['gmv_3']
        age_3 = pd_filter_gmv_3['age_3']
        test_gmv_3 = gmv_3.filter(items=cv_index, axis=0)
        test_age_3 = age_3.filter(items=cv_index, axis=0)

        train_gmv_3 = pd_age_gmv_sex_eth['gmv_3']
        train_gmv_3.loc[cv_index] = float('nan')
        train_age_3 = pd_age_gmv_sex_eth['age_3']
        train_age_3.loc[cv_index] = float('nan')

        # age_2, age_3, gmv_2, gmv_3
        train_data = pd.concat([pd_age_gmv_sex_eth['age_2'], train_age_3,
                                pd_age_gmv_sex_eth['gmv_2'], train_gmv_3], axis=1)

        imputed_result = mice(train_data.to_numpy())
        print(imputed_result)
        imputed_data = pd.DataFrame(data=imputed_result, index=pd_index, columns=['age_2', 'age_3', 'gmv_2', 'gmv_3'])
        gmv_3 = imputed_data['gmv_3']
        age_3 = imputed_data['age_3']
        pred_gmv_3 = gmv_3.filter(items=cv_index, axis=0)
        pred_age_3 = age_3.filter(items=cv_index, axis=0)

        gmv_rmse_0.append(mse(test_gmv_3, pred_gmv_3, squared=False))
        age_rmse_0.append(mse(test_age_3, pred_age_3, squared=False))

        # age_2, age_3, gmv_2, gmv_3, sex, eth
        train_data = pd.concat([pd_age_gmv_sex_eth['age_2'], train_age_3,
                                pd_age_gmv_sex_eth['gmv_2'], train_gmv_3,
                                pd_age_gmv_sex_eth['sex'], pd_age_gmv_sex_eth['eth_0']], axis=1)

        imputed_result = mice(train_data.to_numpy())
        print(imputed_result)
        imputed_data = pd.DataFrame(data=imputed_result, index=pd_index,
                                    columns=['age_2', 'age_3', 'gmv_2', 'gmv_3', 'sex', 'eth_0'])
        gmv_3 = imputed_data['gmv_3']
        age_3 = imputed_data['age_3']
        pred_gmv_3 = gmv_3.filter(items=cv_index, axis=0)
        pred_age_3 = age_3.filter(items=cv_index, axis=0)

        gmv_rmse_1.append(mse(test_gmv_3, pred_gmv_3, squared=False))
        age_rmse_1.append(mse(test_age_3, pred_age_3, squared=False))
        break

    print('######## GMV, age ########')
    print('RMSE of GMV: ', np.round(gmv_rmse_0, 3))
    print('RMSE of age: ', np.round(age_rmse_0, 3), '\n')

    print('######## GMV, age, sex, eth ########')
    print('RMSE of GMV: ', np.round(gmv_rmse_1, 3))
    print('RMSE of age: ', np.round(age_rmse_1, 3), '\n')



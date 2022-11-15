import numpy as np
import pandas as pd
from pd_data_preprocess import Pandas_data
from stat_utils import Stat_utils
from sklearn.metrics import mean_squared_error as mse
from impyute.imputation.cs import mice


def rrmse(y_true, y_pred):
    error = mse(y_true, y_pred, squared=True)
    error = error / np.sum(np.power(y_pred, 2))
    r_rmse = np.sqrt(error)
    return r_rmse


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
    pd_filter_age_3 = pd_age_gmv_sex_eth.dropna(subset=['age_3'])
    pd_index_gmv_3 = pd_filter_gmv_3.index
    pd_index_age_3 = pd_filter_age_3.index
    cv_gmv_length = int(len(pd_index_gmv_3) / 10)
    cv_age_length = int(len(pd_index_age_3) / 10)

    gmv_rmse = []
    age_rmse = []
    both_gmv_rmse = []
    both_age_rmse = []
    gmv_rrmse = []
    age_rrmse = []
    both_gmv_rrmse = []
    both_age_rrmse = []

    for i in range(10):
        if i == 9:
            cv_gmv_index = pd_index_gmv_3[(i * cv_gmv_length):]
            cv_age_index = pd_index_age_3[(i * cv_age_length):]
        else:
            cv_gmv_index = pd_index_gmv_3[(i * cv_gmv_length):((i + 1) * cv_gmv_length)]
            cv_age_index = pd_index_age_3[(i * cv_age_length):((i + 1) * cv_age_length)]

        gmv_3 = pd_filter_gmv_3['gmv_3']
        age_3 = pd_filter_age_3['age_3']
        test_gmv_3 = gmv_3.filter(items=cv_gmv_index, axis=0)
        test_age_3 = age_3.filter(items=cv_age_index, axis=0)

        train_gmv_3 = pd_age_gmv_sex_eth['gmv_3'].copy()
        train_gmv_3.loc[cv_gmv_index] = float('nan')
        train_age_3 = pd_age_gmv_sex_eth['age_3'].copy()
        train_age_3.loc[cv_age_index] = float('nan')

        # only remove gmv
        train_data = pd.concat([pd_age_gmv_sex_eth['age_2'], pd_age_gmv_sex_eth['age_3'],
                                pd_age_gmv_sex_eth['gmv_2'], train_gmv_3,
                                pd_age_gmv_sex_eth['sex'], pd_age_gmv_sex_eth['eth_0']], axis=1)
        imputed_result = mice(train_data.to_numpy())
        imputed_data = pd.DataFrame(data=imputed_result, index=pd_index,
                                    columns=['age_2', 'age_3', 'gmv_2', 'gmv_3', 'sex', 'eth_0'])

        gmv_3 = imputed_data['gmv_3']
        pred_gmv_3 = gmv_3.filter(items=cv_gmv_index, axis=0)

        gmv_rmse.append(mse(test_gmv_3, pred_gmv_3, squared=False))
        gmv_rrmse.append(rrmse(test_gmv_3, pred_gmv_3))

        # only remove age
        train_data = pd.concat([pd_age_gmv_sex_eth['age_2'], train_age_3,
                                pd_age_gmv_sex_eth['gmv_2'], pd_age_gmv_sex_eth['gmv_3'],
                                pd_age_gmv_sex_eth['sex'], pd_age_gmv_sex_eth['eth_0']], axis=1)
        imputed_result = mice(train_data.to_numpy())
        imputed_data = pd.DataFrame(data=imputed_result, index=pd_index,
                                    columns=['age_2', 'age_3', 'gmv_2', 'gmv_3', 'sex', 'eth_0'])

        age_3 = imputed_data['age_3']
        pred_age_3 = age_3.filter(items=cv_age_index, axis=0)

        age_rmse.append(mse(test_age_3, pred_age_3, squared=False))
        age_rrmse.append(rrmse(test_age_3, pred_age_3))

        # remove both gmv and age
        train_data = pd.concat([pd_age_gmv_sex_eth['age_2'], train_age_3,
                                pd_age_gmv_sex_eth['gmv_2'], train_gmv_3,
                                pd_age_gmv_sex_eth['sex'], pd_age_gmv_sex_eth['eth_0']], axis=1)
        imputed_result = mice(train_data.to_numpy())
        imputed_data = pd.DataFrame(data=imputed_result, index=pd_index,
                                    columns=['age_2', 'age_3', 'gmv_2', 'gmv_3', 'sex', 'eth_0'])

        gmv_3 = imputed_data['gmv_3']
        pred_gmv_3 = gmv_3.filter(items=cv_gmv_index, axis=0)

        both_gmv_rmse.append(mse(test_gmv_3, pred_gmv_3, squared=False))
        both_gmv_rrmse.append(rrmse(test_gmv_3, pred_gmv_3))

        age_3 = imputed_data['age_3']
        pred_age_3 = age_3.filter(items=cv_age_index, axis=0)

        both_age_rmse.append(mse(test_age_3, pred_age_3, squared=False))
        both_age_rrmse.append(rrmse(test_age_3, pred_age_3))

    print('######## GMV Imputation ########')
    print('RMSE of GMV: ', np.round(gmv_rmse, 3))
    print('Average: ', np.average(gmv_rmse), '\n')
    print('RRMSE of GMV: ', np.round(gmv_rrmse, 3))
    print('Average: ', np.average(gmv_rrmse), '\n\n')

    print('######## Age Imputation ########')
    print('RMSE of age: ', np.round(age_rmse, 3))
    print('Average: ', np.average(age_rmse), '\n')
    print('RRMSE of age: ', np.round(age_rrmse, 3))
    print('Average: ', np.average(age_rrmse), '\n\n')

    print('######## GMV & Age Imputation ########')
    print('RMSE of GMV: ', np.round(both_gmv_rmse, 3))
    print('Average: ', np.average(both_gmv_rmse), '\n')
    print('RMSE of age: ', np.round(both_age_rmse, 3))
    print('Average: ', np.average(both_age_rmse), '\n')
    print('RRMSE of GMV: ', np.round(both_gmv_rrmse, 3))
    print('Average: ', np.average(both_gmv_rrmse), '\n')
    print('RRMSE of age: ', np.round(both_age_rrmse, 3))
    print('Average: ', np.average(both_age_rrmse), '\n')

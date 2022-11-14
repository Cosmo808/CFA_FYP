import numpy as np
import pandas as pd
from pd_data_preprocess import Pandas_data
from va_baye_gaus_mix import BGM
from stat_utils import Stat_utils
from plot_utils import Plot_utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter
import math
import os
import impyute.imputation.cs.mice as mice
import statsmodels.formula.api as smf


def save_np(file_name, np_array):
    np_array_path = 'data/age_gmv_imputation'
    if not os.path.exists(np_array_path):
        os.makedirs(np_array_path)
    file_name = os.path.join(np_array_path, file_name)
    np.save(file_name, np_array)


if __name__ == "__main__":
    imputation_flag = False
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

    np_age_gmv = pd_age_gmv_sex_eth.iloc[:, :4].to_numpy()
    if imputation_flag:
        imputed_data = mice(np_age_gmv)
        save_np('cov_imputation', imputed_data)
    imputed_data = np.load('data/age_gmv_imputation/cov_imputation.npy')
    pd_imputed_age_gmv = pd.DataFrame(data=imputed_data, index=pd_index, columns=['age_2', 'age_3', 'gmv_2', 'gmv_3'])
    pd_imputed_data = pd.concat([pd_imputed_age_gmv, pd_age_gmv_sex_eth.iloc[:, 4:]], axis=1)

    # expand age
    pd_imputed_age_0 = pd_imputed_data['age_2']
    pd_imputed_age_1 = pd_imputed_data['age_3']
    ex_imputed_age = pd.concat([pd_imputed_age_0, pd_imputed_age_1])

    # expand gmv
    pd_imputed_gmv_0 = pd_imputed_data['gmv_2']
    pd_imputed_gmv_1 = pd_imputed_data['gmv_3']
    ex_imputed_gmv = pd.concat([pd_imputed_gmv_0, pd_imputed_gmv_1])

    # expand sex
    pd_imputed_sex = pd_imputed_data['sex']
    ex_imputed_sex = pd.concat([pd_imputed_sex, pd_imputed_sex])

    # expand ethnicity
    pd_imputed_eth = pd_imputed_data['eth_0']
    ex_imputed_eth = pd.concat([pd_imputed_eth, pd_imputed_eth])

    # expand time point
    time_point_0 = pd.DataFrame(data=np.zeros_like(pd_imputed_gmv_0.to_numpy()), index=pd_index, columns=['time_point'])
    time_point_1 = pd.DataFrame(data=np.ones_like(pd_imputed_gmv_1.to_numpy()), index=pd_index, columns=['time_point'])
    ex_time_point = pd.concat([time_point_0, time_point_1])

    # total expanded data
    pd_ex_data = pd.concat([ex_imputed_age, ex_imputed_gmv, ex_time_point, ex_imputed_sex, ex_imputed_eth], axis=1)
    pd_ex_data = pd_ex_data.rename(columns={0: 'age', 1: 'gmv', 'eth_0': 'ethnicity'})

    # regress out covariates
    fixed_intercept = 1046755.251
    fixed_age_slope = -3874.538
    fixed_sex_slope = -23596.208
    fixed_eth_slope = 3729.030

    random_intercept = 736388381.171
    random_time_point_slope = 41077754.234
    cov_inter_slope = -91070300.594

    regressed_gmv = pd_ex_data['gmv'] - (fixed_age_slope * pd_ex_data['age'] + fixed_sex_slope * pd_ex_data['sex']
                                         + fixed_eth_slope * pd_ex_data['ethnicity'])

    len_index = len(pd_index.to_numpy())
    pd_imputed_data['reg_gmv_0'] = regressed_gmv[:len_index]
    pd_imputed_data['reg_gmv_1'] = regressed_gmv[len_index:]
    # print(pd_imputed_da  ta)

    regressed_baseline = pd_imputed_data['reg_gmv_0']
    per_year = np.average(pd_imputed_data['age_3'] - pd_imputed_data['age_2'])
    regressed_slope = (pd_imputed_data['reg_gmv_1'] - pd_imputed_data['reg_gmv_0']) / per_year

    params = stat.linear_regression_params(regressed_baseline.to_numpy().reshape(-1, 1), regressed_slope)
    intercept = params['Coefficients'][0]
    slope = params['Coefficients'][1]

    plt.figure(0)
    for x, y in zip(regressed_baseline, regressed_slope):
        if np.random.rand() < 0.01:
            plt.scatter(x, y)
    predicted_slope = regressed_baseline * slope + intercept
    plt.plot(regressed_baseline, predicted_slope)
    plt.xlabel('random intercept (baseline status)')
    plt.ylabel('random slope (rate of change)')
    plt.xlim([np.min(regressed_baseline), np.max(regressed_baseline)])
    plt.ylim([np.min(predicted_slope), np.max(predicted_slope)])

    plt.figure(1)
    start = np.max(regressed_baseline)
    time = np.linspace(0, 200, 500)
    for i in range(len(time)):
        if time[i] == 200:
            break
        t_slope = start * slope + intercept
        end = start + t_slope * (time[i+1] - time[i])
        if (start - end) / start < 1e-05:
            break
        plt.plot([time[i], time[i+1]], [start, end], 'black')
        plt.xlabel('time/year')
        plt.ylabel('GMV after regressing out covariates')
        plt.title('Progression of GMV')
        start = end
    plt.show()

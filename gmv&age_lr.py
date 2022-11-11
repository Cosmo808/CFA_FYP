import numpy as np
import pandas as pd
import impyute.imputation.cs.mice as mice
from stat_utils import Stat_utils
from pd_data_preprocess import Pandas_data
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.regression import mixed_linear_model
import os


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

    # age_0: 54243, age_1: 5305
    # age_0 & age_1: 5305

    # gmv_0: 42806, gmv_1: 4622
    # gmv_0 & gmv_1: 4485

    pd_age_gmv = pd.concat([pd_age, pd_gmv], axis=1).dropna(subset=['gmv_2', 'age_2'])    # 42806
    pd_index = pd_age_gmv.index
    np_age_gmv = pd_age_gmv.to_numpy()

    if imputation_flag:
        imputed_data = mice(np_age_gmv)
        save_np('imputation', imputed_data)
    imputed_data = np.load('data/age_gmv_imputation/imputation.npy')
    pd_imputed_data = pd.DataFrame(data=imputed_data, index=pd_index, columns=['age_2', 'age_3', 'gmv_2', 'gmv_3'])

    # expand age
    pd_imputed_age_0 = pd_imputed_data.iloc[:, 0]
    pd_imputed_age_1 = pd_imputed_data.iloc[:, 1]
    ex_imputed_age = pd.concat([pd_imputed_age_0, pd_imputed_age_1])

    # expand gmv
    pd_imputed_gmv_0 = pd_imputed_data.iloc[:, 2]
    pd_imputed_gmv_1 = pd_imputed_data.iloc[:, 3]
    ex_imputed_gmv = pd.concat([pd_imputed_gmv_0, pd_imputed_gmv_1])

    # expand time point
    time_point_0 = pd.DataFrame(data=np.zeros_like(pd_imputed_gmv_0.to_numpy()), index=pd_index, columns=['time_point'])
    time_point_1 = pd.DataFrame(data=np.ones_like(pd_imputed_gmv_1.to_numpy()), index=pd_index, columns=['time_point'])
    ex_time_point = pd.concat([time_point_0, time_point_1])
    pd_ex_data = pd.concat([ex_imputed_age, ex_imputed_gmv, ex_time_point], axis=1)
    pd_ex_data = pd_ex_data.rename(columns={0: 'age', 1: 'gmv'})

    baseline_gmv = pd_imputed_data['gmv_2'].to_numpy()
    slope_gmv = (pd_imputed_data['gmv_3'] - pd_imputed_data['gmv_2']).to_numpy()
    # plt.scatter(baseline_gmv[:1000], slope_gmv[:1000])
    # plt.show()

    # gmv_it = (b_0 + b_1 * age_it) + (B_0i + B_1i * t)
    me_model = smf.mixedlm('gmv ~ age', data=pd_ex_data, groups=pd_ex_data.index, re_formula='~time_point')
    free = mixed_linear_model.MixedLMParams.from_components(np.ones(2), np.eye(2))
    me_model = me_model.fit(free=free, method=['cg', 'lbfgs'])
    print(me_model.summary())


import os
import numpy as np
import pandas as pd
from pd_data_preprocess import Pandas_data
from stat_utils import Stat_utils
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

    pd_age_gmv_sex_eth = pd.concat([pd_age, pd_gmv, pd_sex, pd_eth], axis=1)    # 502411
    pd_age_gmv_sex_eth = pd_age_gmv_sex_eth.dropna(subset=['gmv_2'])    # 42806
    pd_age_gmv_sex_eth = pd_age_gmv_sex_eth.dropna(subset=['sex'])    # 41766
    pd_age_gmv_sex_eth = pd_age_gmv_sex_eth.dropna(subset=['eth_0'])    # 41758

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
    print(pd_ex_data)

    me_model = smf.mixedlm('gmv ~ time_point + sex + ethnicity', data=pd_ex_data, groups=pd_ex_data.index, re_formula='~time_point')
    me_model = me_model.fit(method=['lbfgs', 'cg'])
    print(me_model.summary())

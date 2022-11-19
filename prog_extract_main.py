from progression_feature_extraction import Prog_feat_extract
from pd_data_preprocess import Pandas_data
from stat_utils import Stat_utils
from statsmodels.api import load
import impyute.imputation.cs.mice as mice
import numpy as np
import pandas as pd
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
    pd_sex = data.sex
    pd_eth = data.eth

    pd_age_gmv_sex_eth = pd.concat([pd_age, pd_gmv, pd_sex, pd_eth], axis=1)  # 502411
    pd_age_gmv_sex_eth = pd_age_gmv_sex_eth.dropna(subset=['gmv_2'])  # 42806
    pd_age_gmv_sex_eth = pd_age_gmv_sex_eth.dropna(subset=['sex'])  # 41766
    pd_age_gmv_sex_eth = pd_age_gmv_sex_eth.dropna(subset=['eth_0'])  # 41758
    pd_index = pd_age_gmv_sex_eth.index

    if imputation_flag:
        imputed_data = mice(pd_age_gmv_sex_eth.to_numpy())
        save_np('cov_imputation', imputed_data)
    imputed_data = np.load('data/age_gmv_imputation/cov_imputation.npy')
    pd_imputed_data = pd.DataFrame(data=imputed_data, index=pd_index,
                                   columns=['age_2', 'age_3', 'gmv_2', 'gmv_3', 'sex', 'eth_0'])

    # expand age: age_ij
    pd_imputed_age_0 = pd_imputed_data['age_2']
    pd_imputed_age_1 = pd_imputed_data['age_3']
    ex_imputed_age = pd.concat([pd_imputed_age_0, pd_imputed_age_1])

    # expand delta age: age_ij - age_i0
    ex_imputed_baseline_age = pd.concat([pd_imputed_age_0, pd_imputed_age_0])
    ex_imputed_delta_age = ex_imputed_age - ex_imputed_baseline_age

    # expand gmv: gmv_ij
    pd_imputed_gmv_0 = pd_imputed_data['gmv_2']
    pd_imputed_gmv_1 = pd_imputed_data['gmv_3']
    ex_imputed_gmv = pd.concat([pd_imputed_gmv_0, pd_imputed_gmv_1])

    # expand baseline gmv: gmv_i0
    ex_imputed_baseline_gmv = pd.concat([pd_imputed_gmv_0, pd_imputed_gmv_0])

    # expand sex
    pd_imputed_sex = pd_imputed_data['sex']
    ex_imputed_sex = pd.concat([pd_imputed_sex, pd_imputed_sex])

    # expand ethnicity
    pd_imputed_eth = pd_imputed_data['eth_0']
    ex_imputed_eth = pd.concat([pd_imputed_eth, pd_imputed_eth])

    # total expanded data
    pd_ex_data = pd.concat([ex_imputed_age, ex_imputed_delta_age, ex_imputed_gmv,
                            ex_imputed_baseline_gmv, ex_imputed_sex, ex_imputed_eth], axis=1)
    pd_ex_data = pd_ex_data.rename(columns={0: 'age', 1: 'delta_age', 2: 'gmv',
                                            'gmv_2': 'gmv_0', 'eth_0': 'ethnicity'})

    me_model = load('model/gmv&age_lme_model/lme_model')
    params = me_model.params

    # regress out covariates
    # reg_gmv = (B_0 + B_1 * delta_age) + b_0
    reg_gmv = pd_ex_data['gmv'] - (params['sex'] * pd_ex_data['sex'] + params['ethnicity'] * pd_ex_data['ethnicity']
                                   + params['age'] * pd_ex_data['age'])
    reg_gmv_0 = reg_gmv.iloc[:int(len(reg_gmv) / 2)]
    reg_gmv_1 = reg_gmv.iloc[int(len(reg_gmv) / 2):]

    data = pd.concat([pd_imputed_age_0, pd_imputed_age_1, reg_gmv_0, reg_gmv_1], axis=1)
    data = data.rename(columns={'age_2': 0, 'age_3': 1, 0: 2, 1: 3})

    prog = Prog_feat_extract(data, 1e-2, 50, 1e-7, 1000)
    prog.prog_iter(threshold=1000)
    prog.prog_extract()

import numpy as np
import pandas as pd
from pd_data_preprocess import Pandas_data
from va_baye_gaus_mix import BGM
from stat_utils import Stat_utils
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

    if imputation_flag:
        imputed_data = mice(pd_age_gmv_sex_eth.to_numpy())
        save_np('cov_imputation', imputed_data)
    imputed_data = np.load('data/age_gmv_imputation/cov_imputation.npy')
    pd_imputed_data = pd.DataFrame(data=imputed_data, index=pd_index,
                                   columns=['age_2', 'age_3', 'gmv_2', 'gmv_3', 'sex', 'eth_0'])

    np.random.sample(pd_index)
    print(pd_index)


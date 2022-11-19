import pandas as pd
import numpy as np
from scipy.special import comb


class Prog_feat_extract:
    def __init__(self, data, sample_rate, neighbor_num, convergence):
        self.data = data
        self.prog_extract_data = data

        self.sample_rate = sample_rate
        self.neighbor_num = neighbor_num
        self.convergence = convergence

    def prog_extract(self):
        index = self.data.index
        sample_num = int(self.sample_rate * len(index))
        sample_permutation = np.random.permutation(np.arange(len(index)))
        sample_index = index[sample_permutation[:sample_num]]
        sample_point = self.data.filter(items=sample_index, axis=0)

        for i in range(sample_num):
            center = sample_point.iloc[i, :]
            center_x_0 = center[0]
            center_x_1 = center[1]
            center_y_0 = center[2]
            center_y_1 = center[3]
            neighbor_radius = ((center_x_0 - center_x_1) ** 2 + (center_y_0 - center_y_1) ** 2) ** 0.5 / 150

            d_00 = ((self.data.iloc[:, 0] - center_x_0) ** 2 + (self.data.iloc[:, 2] - center_y_0) ** 2) ** 0.5
            d_01 = ((self.data.iloc[:, 0] - center_x_1) ** 2 + (self.data.iloc[:, 2] - center_y_1) ** 2) ** 0.5
            d_10 = ((self.data.iloc[:, 1] - center_x_0) ** 2 + (self.data.iloc[:, 3] - center_y_0) ** 2) ** 0.5
            d_11 = ((self.data.iloc[:, 1] - center_x_1) ** 2 + (self.data.iloc[:, 3] - center_y_1) ** 2) ** 0.5

            neighbor_d = pd.concat([d_00, d_01, d_10, d_11], axis=1)
            neighbor_d_min = neighbor_d.min(axis=1)
            neighbor_d = neighbor_d_min[neighbor_d_min <= neighbor_radius]
            neighbor_index = neighbor_d.index

            # no neighbor
            if len(neighbor_index) == 1:
                continue

            # if exceed maximum, select nearest points
            neighbor_len = len(neighbor_index)
            if neighbor_len > self.neighbor_num:
                neighbor_len = self.neighbor_num
                neighbor_d = neighbor_d.nsmallest(neighbor_len)

            # find the neighbor minimizing the error
            neighbor_index, neighbor_centroid = self.minimize_error(center, neighbor_d)

            self.data = self.data.drop(neighbor_index)
            self.data = self.data.append(neighbor_centroid)
        self.prog_extract_data = pd.concat([self.prog_extract_data, self.data], axis=1)
        print(self.prog_extract_data)

    def minimize_error(self, center, neighbor_d):
        center_x_0 = center[0]
        center_x_1 = center[1]
        center_y_0 = center[2]
        center_y_1 = center[3]

        index = neighbor_d.index
        num = len(index)
        all_list = np.arange(num)
        iter_cnt = 0

        while num > 0:
            select_list = all_list[:num]
            select_index = index[select_list]
            select_neighbor = neighbor_d.filter(items=select_index, axis=0)

            neighbor_weight = np.exp(
                -select_neighbor ** 2 / (2 * (select_neighbor.max() - select_neighbor.min()) ** 2)
            ) / num

            neighbor_point = self.data.filter(items=select_index, axis=0)
            neighbor_centroid_x_0 = np.sum(neighbor_point.iloc[:, 0] * neighbor_weight)
            neighbor_centroid_x_1 = np.sum(neighbor_point.iloc[:, 1] * neighbor_weight)
            neighbor_centroid_y_0 = np.sum(neighbor_point.iloc[:, 2] * neighbor_weight)
            neighbor_centroid_y_1 = np.sum(neighbor_point.iloc[:, 3] * neighbor_weight)

            center_vec = [center_x_1 - center_x_0, center_y_1 - center_y_0]
            neighbor_vec = [neighbor_centroid_x_1 - neighbor_centroid_x_0,
                            neighbor_centroid_y_1 - neighbor_centroid_y_0]
            cos = (center_vec[0] * neighbor_vec[0] + center_vec[1] * neighbor_vec[1]) / \
                  ((center_vec[0] ** 2 + center_vec[1] ** 2) * (neighbor_vec[0] ** 2 + neighbor_vec[1] ** 2)) ** 0.5

            if (1 - cos) < self.convergence:
                neighbor_centroid = pd.DataFrame(
                    [[neighbor_centroid_x_0, neighbor_centroid_x_1,
                      neighbor_centroid_y_0, neighbor_centroid_y_1]], index=[select_index[0]]
                )
                return select_index, neighbor_centroid
            else:
                iter_cnt += 1
                if iter_cnt >= min(10000, comb(len(index), num)):
                    iter_cnt = 0
                    num -= 1
                    continue

                # metropolis-hastings
                if iter_cnt == 1:
                    last_error = 1 - cos
                    last_list = all_list.copy()
                    change_index_0 = np.random.permutation(num - 1)[0] + 1
                    change_index_1 = np.random.permutation(len(index) - num)[0] + num
                    all_list[change_index_0], all_list[change_index_1] = all_list[change_index_1], all_list[change_index_0]
                else:
                    alpha = last_error / (1 - cos)
                    if np.random.rand() * 0.2 + 0.8 < alpha:
                        last_error = 1 - cos
                        last_list = all_list
                        change_index_0 = np.random.permutation(num)[0]
                        change_index_1 = np.random.permutation(len(index) - num)[0] + num
                        all_list[change_index_0], all_list[change_index_1] = all_list[change_index_1], all_list[change_index_0]
                    else:
                        all_list = last_list





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
from statsmodels.api import load
import impyute.imputation.cs.mice as mice
from progression_feature_extraction import Prog_feat_extract
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
    prog = Prog_feat_extract(data, 1e-3, 50, 1e-10)
    prog.prog_extract()

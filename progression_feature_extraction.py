import pandas as pd
import numpy as np
import os
from scipy.special import comb


class Prog_feat_extract:
    def __init__(self, data, sample_rate, neighbor_num, convergence, max_iterations):
        self.data = data
        self.prog_extract_data = data

        self.sample_rate = sample_rate
        self.neighbor_num = neighbor_num
        self.convergence = convergence
        self.max_iter = max_iterations

    def prog_extract(self):
        index = self.data.index
        sample_num = int(self.sample_rate * len(index))
        sample_permutation = np.random.permutation(np.arange(len(index)))
        sample_index = index[sample_permutation[:sample_num]]
        sample_point = self.data.filter(items=sample_index, axis=0)
        print('######## Sample number: {} ########'.format(sample_num))
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
            if i % 10 == 1:
                print('######## Progressing: {}% ({} / {})'.format(np.round(i / sample_num * 100, 2), i, sample_num))
                print('######## Number of data: {}'.format(len(self.data.index)))
        self.prog_extract_data = pd.concat([self.prog_extract_data, self.data], axis=1)

    def minimize_error(self, center, neighbor_d):
        center_x_0 = center[0]
        center_x_1 = center[1]
        center_y_0 = center[2]
        center_y_1 = center[3]

        index = neighbor_d.index
        num = len(index)
        all_list = np.arange(num)
        iter_cnt = 0
        last_error = 0
        last_list = []

        while num > 0:
            if num == 1:
                return index, center
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
            if (center_vec[0] ** 2 + center_vec[1] ** 2) * (neighbor_vec[0] ** 2 + neighbor_vec[1] ** 2) == 0:
                cos = 0
            else:
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
                if iter_cnt >= min(self.max_iter, comb(len(index), num)):
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

    def prog_iter(self, threshold):
        if not os.path.exists('data/prog_feature_extract'):
            os.makedirs('data/prog_feature_extract')
        iteration = 0
        while True:
            print('\n###################################\n########## Iteration {:^3} ##########\n'
                  '###################################'.format(iteration))
            self.prog_extract()
            self.data.to_csv(path_or_buf='data/prog_feature_extract/iter_{}.csv'.format(iteration))
            iteration += 1
            if len(self.data.index) < threshold:
                self.prog_extract_data.to_csv(path_or_buf='data/prog_feature_extract/all_data.csv')
                break

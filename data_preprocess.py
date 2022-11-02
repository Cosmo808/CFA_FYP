import numpy as np
import os


class Data:
    def __init__(self, data_path='./data/np_array'):
        # data
        self.age = np.load(os.path.join(data_path, 'age.npy'))
        self.bfp = np.load(os.path.join(data_path, 'bfp.npy'))
        self.bmi = np.load(os.path.join(data_path, 'bmi.npy'))
        self.eid = np.load(os.path.join(data_path, 'eid.npy'))    # string
        self.sex = np.load(os.path.join(data_path, 'sex.npy'))    # string
        
        # pairwise
        self.nz_cnt = None
        self.pair_ind = None

    def pairwise_data(self):
        nz_cnt_age = []
        nz_cnt_bfp = []
        nz_cnt_bmi = []
        pair_ind_age = []
        pair_ind_bfp = []
        pair_ind_bmi = []
        for i, j, k in zip(self.age, self.bfp, self.bmi):
            nz_cnt_age.append(np.count_nonzero(i))
            nz_cnt_bfp.append(np.count_nonzero(j))
            nz_cnt_bmi.append(np.count_nonzero(k))
            
            if nz_cnt_age[-1] >= 2:
                pair_ind_age.append(len(nz_cnt_age)-1)
            if nz_cnt_bfp[-1] >= 2:
                pair_ind_bfp.append(len(nz_cnt_bfp)-1)
            if nz_cnt_bmi[-1] >= 2:
                pair_ind_bmi.append(len(nz_cnt_bmi)-1)
                
        self.nz_cnt = [nz_cnt_age, nz_cnt_bfp, nz_cnt_bmi]
        self.pair_ind = [pair_ind_age, pair_ind_bfp, pair_ind_bmi]

        pairwise_age = self.obtain_pairwise_data(self.age, pair_ind_age)
        pairwise_bfp = self.obtain_pairwise_data(self.bfp, pair_ind_bfp)
        pairwise_bmi = self.obtain_pairwise_data(self.bmi, pair_ind_bmi)

        return pairwise_age, pairwise_bfp, pairwise_bmi

    @staticmethod
    def obtain_pairwise_data(data_input, index):
        pairwise_data = []
        for ind in index:
            data_row = data_input[ind]
            data_nz = data_row[data_row != 0]
            for i in range(len(data_nz)-1):
                data_pair = [data_nz[i], data_nz[i+1], ind]
                pairwise_data.append(data_pair)
        return np.array(pairwise_data)

    @staticmethod
    def combine_pairwise_data(pair_data1, pair_data2):
        index1 = pair_data1[:, -1]
        index2 = pair_data2[:, -1]
        index_eq = []
        for i, ind in enumerate(index1):
            eq = np.where(index2 == ind)[0]
            if len(eq) >= 1:
                index_eq.append([i, eq[0]])

        combined_data1 = []
        combined_data2 = []
        for ind in index_eq:
            ind1 = ind[0]
            ind2 = ind[1]
            combined_data1.append(pair_data1[ind1])
            combined_data2.append(pair_data2[ind2])
        return np.array(combined_data1), np.array(combined_data2)



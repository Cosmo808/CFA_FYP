import numpy as np
import os


class Data:
    def __init__(self, data_path='./data/np_array'):
        # data
        self.age = np.load(os.path.join(data_path, 'age.npy'))
        self.bfp = np.load(os.path.join(data_path, 'bfp.npy'))
        self.bmi = np.load(os.path.join(data_path, 'bmi.npy'))
        self.eid = np.load(os.path.join(data_path, 'eid.npy'))    # string
        self.gmv = np.load(os.path.join(data_path, 'gmv.npy'))
        self.sex = np.load(os.path.join(data_path, 'sex.npy'))    # string
        
        # pairwise
        self.nz_cnt = None
        self.pair_ind = None

    def pairwise_data(self):
        nz_cnt_age = []
        nz_cnt_bfp = []
        nz_cnt_bmi = []
        nz_cnt_gmv = []
        pair_ind_age = []
        pair_ind_bfp = []
        pair_ind_bmi = []
        pair_ind_gmv = []
        for i, j, k, l in zip(self.age, self.bfp, self.bmi, self.gmv):
            nz_cnt_age.append(np.count_nonzero(i))
            nz_cnt_bfp.append(np.count_nonzero(j))
            nz_cnt_bmi.append(np.count_nonzero(k))
            nz_cnt_gmv.append(np.count_nonzero(l))
            
            if nz_cnt_age[-1] >= 2:
                pair_ind_age.append(len(nz_cnt_age)-1)
            if nz_cnt_bfp[-1] >= 2:
                pair_ind_bfp.append(len(nz_cnt_bfp)-1)
            if nz_cnt_bmi[-1] >= 2:
                pair_ind_bmi.append(len(nz_cnt_bmi)-1)
            if nz_cnt_gmv[-1] >= 2:
                pair_ind_gmv.append(len(nz_cnt_gmv)-1)
                
        self.nz_cnt = [nz_cnt_age, nz_cnt_bfp, nz_cnt_bmi, nz_cnt_gmv]
        self.pair_ind = [pair_ind_age, pair_ind_bfp, pair_ind_bmi, pair_ind_gmv]

        pairwise_age = self.obtain_pairwise_data(self.age, pair_ind_age)
        pairwise_bfp = self.obtain_pairwise_data(self.bfp, pair_ind_bfp)
        pairwise_bmi = self.obtain_pairwise_data(self.bmi, pair_ind_bmi)
        pairwise_gmv = self.obtain_pairwise_data(self.gmv, pair_ind_gmv)

        return pairwise_age, pairwise_bfp, pairwise_bmi, pairwise_gmv

    @staticmethod
    def obtain_pairwise_data(data_input, index):
        pairwise_data = []
        for ind in index:
            data_row = data_input[ind]
            data_nz = data_row[data_row != 0]
            ind_nz = np.where(data_row != 0)[0]
            for i in range(len(data_nz)-1):
                data_pair = [data_nz[i], data_nz[i+1], ind, ind_nz[i], ind_nz[i+1]]
                pairwise_data.append(data_pair)
        return np.array(pairwise_data, dtype=float)

    @staticmethod
    def combine_pairwise_data(pair_data1, pair_data2):
        index1 = pair_data1[:, 2]
        index2 = pair_data2[:, 2]
        first1 = pair_data1[:, 3]
        first2 = pair_data2[:, 3]
        second1 = pair_data1[:, 4]
        second2 = pair_data2[:, 4]

        eq_cnt1 = []
        eq_cnt2 = []

        for cnt1, ind, fir, sec in zip(range(len(index1)), index1, first1, second1):
            eq_index = np.where(index2 == ind)[0]
            fir2s = first2[eq_index]
            try:
                eq_first = np.where(fir2s == fir)[0][0]
            except IndexError:
                continue
            sec2 = second2[eq_index][eq_first]
            if sec == sec2:
                eq_cnt1.append(cnt1)
                cnt2 = eq_index[eq_first]
                eq_cnt2.append(cnt2)

        combine_data1 = []
        combine_data2 = []
        for cnt in eq_cnt1:
            combine_data1.append(pair_data1[cnt])
        for cnt in eq_cnt2:
            combine_data2.append(pair_data2[cnt])
        return np.array(combine_data1), np.array(combine_data2)

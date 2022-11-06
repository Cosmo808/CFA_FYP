import numpy as np
from np_data_preprocess import Numpy_data
from stat_utils import Stat_utils
import os


def save_np(file_name, np_array):
    np_array_path = 'data/data_age_bfp_gmv'
    if not os.path.exists(np_array_path):
        os.makedirs(np_array_path)
    file_name = os.path.join(np_array_path, file_name)
    np.save(file_name, np_array)


if __name__ == "__main__":
    data = Numpy_data()
    stat = Stat_utils()
    match_flag = False

    if match_flag:
        pairwise_age, pairwise_bfp, pairwise_bmi, pairwise_gmv = data.pairwise_data()
        single_age, single_bfp, single_bmi, single_gmv = data.single_point_data()

        single_gmv, single_age = data.match_single_data(single_gmv, single_age)
        single_age, single_bfp = data.match_single_data(single_age, single_bfp)
        single_gmv, single_bfp = data.match_single_data(single_gmv, single_bfp)

        save_np('age', single_age)
        save_np('bfp', single_bfp)
        save_np('gmv', single_gmv)

    data_path = 'data/data_age_bfp_gmv'
    single_age = np.load(os.path.join(data_path, 'age.npy'))
    single_bfp = np.load(os.path.join(data_path, 'bfp.npy'))
    single_gmv = np.load(os.path.join(data_path, 'gmv.npy'))

    # longitudinal: gmv ~ B * age_0 + B * bfp_0 + B * time_point * bfp_type
    #                     + b * time_point + b * bfp_type



import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from data_preprocess import Data
from va_baye_gaus_mix import BGM
from plot_utils import Plot_utils
import math


if __name__ == "__main__":
    # hyperparameter
    n_components = 8
    prior = 1e+05

    fit_flag = True
    data = Data()
    bgm = BGM(n_components, prior)
    plot = Plot_utils()

    pairwise_age, pairwise_bfp, pairwise_bmi, pairwise_gmv = data.pairwise_data()
    single_age, single_bfp, single_bmi, single_gmv = data.single_point_data()

    pairwise_gmv, single_bfp = data.match_pairwise_single(pairwise_gmv, single_bfp, 2, 3, 2)

    delta_gmv = pairwise_gmv[:, 1] - pairwise_gmv[:, 0]
    pairwise_data = pairwise_gmv[:, 0:2]
    pairwise_data[:, 0] = single_bfp[:, 0]
    pairwise_data[:, 1] = delta_gmv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from data_preprocess import Data
from va_baye_gaus_mix import BGM
from plot_utils import Plot_utils

if __name__ == "__main__":
    # hyperparameter
    n_components = 5
    prior = 1e+00

    fit_flag = False
    data = Data()
    bgm = BGM(n_components, prior)
    plot = Plot_utils()

    pairwise_age, pairwise_bfp, pairwise_bmi, pairwise_gmv = data.pairwise_data()
    pairwise_age, pairwise_bfp = data.combine_pairwise_data(pairwise_age, pairwise_bfp)


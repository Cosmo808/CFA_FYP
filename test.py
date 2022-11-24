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
from scipy.stats import chi2
from statsmodels.api import load
import impyute.imputation.cs.mice as mice
import statsmodels.formula.api as smf
from scipy import spatial
from scipy.special import comb
from progression_feature_extraction import Prog_feat_extract


def save_np(file_name, np_array):
    np_array_path = 'data/age_gmv_imputation'
    if not os.path.exists(np_array_path):
        os.makedirs(np_array_path)
    file_name = os.path.join(np_array_path, file_name)
    np.save(file_name, np_array)


if __name__ == "__main__":
    me_1 = load('model/gmv&age_lme_model/delta_age+age_0')
    # me_2 = load('model/gmv&age_lme_model/delta_age_2+age_0_2+delta_age')
    #
    # LR_statistic = -2 * (me_1.llf - me_2.llf)
    # p_value = chi2.sf(LR_statistic, 2)

    print(me_1.summary())
    print(me_1.summary().tables[0].iloc[1, 1])
    print(me_1.llf)

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
    a=np.array([0])
    b = np.array([1])
    print((b[0]/a[0])<1)

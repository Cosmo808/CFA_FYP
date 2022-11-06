import numpy as np
import pandas as pd
from pd_data_preprocess import Pandas_data
from va_baye_gaus_mix import BGM
from stat_utils import Stat_utils
from plot_utils import Plot_utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

if __name__ == "__main__":
    # hyperparameter
    n_components = 5
    prior = 1e-00
    max_iter = 3000
    tol = 1e-03

    data = Pandas_data()
    bgm = BGM(n_components, prior, max_iter, tol)
    stat = Stat_utils()
    plot = Plot_utils()

    pd_age = data.age.iloc[:, 2]
    pd_bfp = data.bfp.iloc[:, 2]
    pd_gmv = data.gmv.iloc[:, 0]

    age_bfp_gmv = pd.concat([pd_age, pd_bfp, pd_gmv], axis=1)
    age_bfp_gmv = age_bfp_gmv.dropna()

    a=np.array([1,2,3])
    b=np.array([3,4,5])
    print(np.multiply(a,b))

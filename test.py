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
    a= [1,1,3,3,4,2,1,2,3,4]
    b = [[] for i in range(4)]
    for i in range(1, 5):
        a_np = np.array(a)
        index = np.nonzero(a_np == i)
        index = np.array(index).tolist()[0]
        b[i-1] = index
        print(index)
    print(b)

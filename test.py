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
    a=[1,2,3,4,5,6,7,8]
    print(np.array([a[:2], a[-2:]]).reshape(4))

import numpy as np
import pandas as pd
from pd_data_preprocess import Pandas_data
from va_baye_gaus_mix import BGM
from stat_utils import Stat_utils
from plot_utils import Plot_utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter
import math
import os
import impyute.imputation.cs.mice as mice
import statsmodels.formula.api as smf


if __name__ == "__main__":
    plt.plot([0,5], [8,3])
    plt.show()


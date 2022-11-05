import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from data_preprocess import Data
from va_baye_gaus_mix import BGM
from plot_utils import Plot_utils
import math


if __name__ == "__main__":
    a=np.ones(shape=(3, 2), dtype=float)
    print(a)
    b = np.array([[2,2],[3,4], [5,6]])
    print(np.multiply(a,b))
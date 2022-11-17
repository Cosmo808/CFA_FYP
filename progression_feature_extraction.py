import pandas as pd
import numpy as np


class Prog_feat_extract:
    def __init__(self, data, sample_rate, neighbor_radius, neighbor_num, convergence):
        self.index = data.index
        self.x_0 = data.iloc[:, 0]
        self.x_1 = data.iloc[:, 1]
        self.y_0 = data.iloc[:, 2]
        self.y_1 = data.iloc[:, 3]

        self.sample_rate = sample_rate
        self.neighbor_radius = neighbor_radius
        self.neighbor_num = neighbor_num
        self.convergence = convergence

    def sample_point(self):
        np.random.sample()



import numpy as np
import pandas as pd


class Pandas_data:
    def __init__(self):
        self.age_bfp_bmi_gmv_sex = pd.read_csv('./data/age_bfp_bmi_gmv_sex.csv', index_col=0)
        self.age = self.age_bfp_bmi_gmv_sex.iloc[:, 0:4]
        self.bfp = self.age_bfp_bmi_gmv_sex.iloc[:, 4:8]
        self.bmi = self.age_bfp_bmi_gmv_sex.iloc[:, 8:12]
        self.gmv = self.age_bfp_bmi_gmv_sex.iloc[:, 12:14]
        self.sex = self.age_bfp_bmi_gmv_sex.iloc[:, -1]

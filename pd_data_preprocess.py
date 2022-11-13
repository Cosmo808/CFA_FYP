import pandas as pd


class Pandas_data:
    def __init__(self):
        self.age_bfp_bmi_gmv_sex_eth = pd.read_csv('data/age_bfp_bmi_gmv_sex_eth.csv', index_col=0)
        self.age = self.age_bfp_bmi_gmv_sex_eth.iloc[:, 0:4]
        self.bfp = self.age_bfp_bmi_gmv_sex_eth.iloc[:, 4:8]
        self.bmi = self.age_bfp_bmi_gmv_sex_eth.iloc[:, 8:12]
        self.gmv = self.age_bfp_bmi_gmv_sex_eth.iloc[:, 12:14]
        self.sex = self.age_bfp_bmi_gmv_sex_eth.iloc[:, 14]
        self.eth = self.age_bfp_bmi_gmv_sex_eth.iloc[:, 15]

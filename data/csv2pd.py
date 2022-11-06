import pandas as pd


bmi = pd.read_csv('21001.csv', index_col=0)    # 4
age = pd.read_csv('21003.csv', index_col=0)    # 4
sex = pd.read_csv('22001.csv', index_col=0)    # 1
bfp = pd.read_csv('23099.csv', index_col=0)    # 4
gmv = pd.read_csv('25005.csv', index_col=0)    # 2

data = pd.concat([age, bfp, bmi, gmv, sex], axis=1)
data.to_csv('age_bfp_bmi_gmv_sex.csv')

import pandas as pd


bmi = pd.read_csv('21001.csv', index_col=0)    # 4
age = pd.read_csv('21003.csv', index_col=0)    # 4
sex = pd.read_csv('22001.csv', index_col=0)    # 1
bfp = pd.read_csv('23099.csv', index_col=0)    # 4
gmv = pd.read_csv('25005.csv', index_col=0)    # 2
eth = pd.read_csv('21000.csv', index_col=0, usecols=[0, 1])    # 1
eth = eth.replace([1001, 2001, 3001, 4001], 1)
eth = eth.replace([1003, 1002, 6, 4002, 3002, 3004, -3, 5, 2004, 2003, 2002, 3003, -1, 4003, 2, 3, 4], 0)

data = pd.concat([age, bfp, bmi, gmv, sex, eth], axis=1)
data.to_csv('age_bfp_bmi_gmv_sex_eth.csv')

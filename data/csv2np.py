import numpy as np
import os
import math


def save_np(file_name, np_array):
    np_array_path = './np_array'
    if not os.path.exists(np_array_path):
        os.makedirs(np_array_path)
    file_name = os.path.join(np_array_path, file_name)
    np.save(file_name, np_array)


bmi = np.genfromtxt('21001.csv', delimiter=',', dtype=float)
age = np.genfromtxt('21003.csv', delimiter=',', dtype=int)
sex = np.genfromtxt('22001.csv', delimiter=',', dtype=str)    # male: 1, female: 0
bfp = np.genfromtxt('23099.csv', delimiter=',', dtype=float)    # body fat percentage
gmv = np.genfromtxt('25005.csv', delimiter=',', dtype=float)

# eid and sex
np_eid = []
np_sex = []
for row in sex:
    if row[0] == 'eid':
        continue
    np_eid.append(row[0])
    np_sex.append(row[1])

save_np('eid', np.array(np_eid))
save_np('sex', np.array(np_sex))

# bmi, body fat percentage (bfp) and gmv
np_bmi = []
np_bfp = []
np_gmv = []
for row1, row2, row3 in zip(bmi, bfp, gmv):
    for i, j in enumerate(row1):
        if math.isnan(j):
            row1[i] = 0
    for i, j in enumerate(row2):
        if math.isnan(j):
            row2[i] = 0
    for i, j in enumerate(row3):
        if math.isnan(j):
            row3[i] = 0
    np_bmi.append(row1[1:])
    np_bfp.append(row2[1:])
    np_gmv.append([0, 0, row3[1], row3[2]])

save_np('bmi', np.array(np_bmi))
save_np('bfp', np.array(np_bfp))
save_np('gmv', np.array(np_gmv))

# age
np_age = []
for row in age:
    for i, j in enumerate(row):
        if j == -1:
            row[i] = 0
    np_age.append(row[1:])

save_np('age', np.array(np_age))

import os,sys
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt

feature = sys.argv[1]
genre = sys.argv[2]

label = 'xofy'

file_template = '../../cache/naacl/results/{0}.{1}.fold_{2}.{3}.libsvm.regression.pred'

rmse = []
for fold in range(10):
    pred_file = file_template.format(genre, label, fold, feature)
    arr_1 = []
    arr_2 = []
    with open(pred_file, 'r') as f:
        for line in f.readlines():
            ts = line.strip().split()
            arr_1.append(float(ts[0]))
            arr_2.append(float(ts[1]))
    arr_1 = np.array(arr_1)
    arr_2 = np.array(arr_2)
    v = sqrt(mean_squared_error(arr_1, arr_2))
    rmse.append(v)
print np.mean(rmse)

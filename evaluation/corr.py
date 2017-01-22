import os,sys
import numpy as np
from scipy import stats

feature = sys.argv[1]

if len(sys.argv) > 2:
    genres = sys.argv[2:]
else:
    genres = ['watches', 'cellphone', 'home', 'outdoor', 'electronics']

for genre in genres:
    label = 'xofy'
    #label = 'l2'

    file_template = '../cache/results/{0}.{1}.fold_{2}.{3}.libsvm.regression.pred'
    #file_template = '../cache/results/{0}.{1}.fold_{2}.{3}.maximum.regression.pred'

    pearson = []
    spearman = []
    nnn = 0
    for fold in range(10):
        pred_file = file_template.format(genre, label, fold, feature)
        arr_1 = []
        arr_2 = []
        with open(pred_file, 'r') as f:
            for line in f.readlines():
                ts = line.strip().split()
                arr_1.append(float(ts[0]))
                arr_2.append(float(ts[1]))
                nnn += 1
        arr_1 = np.array(arr_1)
        arr_2 = np.array(arr_2)
        p, pv = stats.pearsonr(arr_1, arr_2)
        s, pv = stats.spearmanr(arr_1, arr_2)
        pearson.append(p)
        spearman.append(s)
        print p, s

    print '{0:.3f}, {1:.3f}, {2}'.format(np.mean(pearson), np.mean(spearman), nnn/10.0)

print "\n"


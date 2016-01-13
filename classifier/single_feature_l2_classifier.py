#! /usr/bin/env python

import os, sys, random,  math
from utils import *
from os.path import *

sys.path.append('../thirdparty/libsvm/python')

from svmutil import *

feature_name = sys.argv[1]

fold_path    = '../cache/folds/'
model_path   = '../cache/models/'
result_path  = '../cache/results/'
develop_path  = '../cache/develop/'
feature_path = '../cache/features/'
score_path   = '../dataset/labels/'

score_type = 'xofy'
overwrite_model = True

asp_dict = {'serv':'service', 'func':'functionality', 'appe':'appearance', 'o':'other',
        'qual':'quality', 'use':'usability', 'price':'price', 'brand':'brand', 'ovrl':'overall'}

genres = sys.argv[2:]

for genre in genres:
    print 'processing {0}'.format(genre)

    for fold_id in fold_ids:
        scores = []; features = []
        for aspect in asp_dict.values():
            fname_dev_res  = '{0}/{1}.{2}.fold_{3}.{4}.libsvm.regression.pred'.format(develop_path, genre, aspect, fold_id, feature_name)
            if not os.path.exists(fname_dev_res):
                continue
            tmp_scores = []; tmp_feats = []
            lines = get_content(fname_dev_res)
            for i in range(len(lines)):
                line = lines[i]; ts = line.strip().split()
                score = float(ts[0]); feat= float(ts[1])
                tmp_scores.append(score); tmp_feats.append(feat)
            if len(features) == 0:
                features = [[f] for f in tmp_feats]
            else:
                for i in range(len(lines)):
                    features[i].append(tmp_feats[i])
            scores = tmp_scores
        y_train = scores; x_train = features
        x_train, y_train = sample_dataset(x_train, y_train, 10000)

        for aspect in asp_dict.values():
            fname_test_res  = '{0}/{1}.{2}.fold_{3}.{4}.libsvm.regression.pred'.format(result_path, genre, aspect, fold_id, feature_name)
            if not os.path.exists(fname_test_res):
                continue
            tmp_scores = []; tmp_feats = []
            lines = get_content(fname_dev_res)
            for i in range(len(lines)):
                line = lines[i]; ts = line.strip().split()
                score = float(ts[0]); feat= float(ts[1])
                tmp_scores.append(score); tmp_feats.append(feat)
            if len(features) == 0:
                features = [[f] for f in tmp_feats]
            else:
                for i in range(len(lines)):
                    features[i].append(tmp_feats[i])
            scores = tmp_scores
        y_test = scores; x_test = features

        ###################################################################
        # classification: training and testing
        model_name = '{0}/{1}.{2}.fold_{3}.{4}.libsvm.regression.model'.format(model_path, genre, 'l2', fold_id, feature_name)
        if overwrite_model or not os.path.exists(model_name):
            mse = [0]*5 # mseuracy
            for cc in range(5):
                tmpC = math.pow(10, cc-2)
                print 'testing C = ', tmpC
                mse[cc] = svm_train(y_train, x_train, '-s 3 -t 2 -v 5 -q -c {0}'.format(tmpC) )
            mse_i = mse.index(min(mse))
            best_c = math.pow(10, mse_i-2)

            print 'The best C is: ', best_c

            m = svm_train(y_train, x_train, '-s 3 -t 2 -q -c {0}'.format(best_c) )
            svm_save_model( model_name, m )
        else:
            best_c = 10
            m = svm_train(y_train, x_train, '-s 3 -t 2 -q -c {0}'.format(best_c) )

        # test set
        y_pred, mse_test, y_pred_2 = svm_predict(y_test, x_test, m)
        out_file = '{0}/{1}.{2}.fold_{3}.{4}.libsvm.regression.pred'.format(result_path, genre, 'l2', fold_id, feature_name)
        fout = open(out_file, 'w+')
        for i in  range(len(y_test)):
            ostr = "{0} {1} {2}\n".format(y_test[i], y_pred[i], y_pred_2[i][0])
            fout.write(ostr)
        fout.close()

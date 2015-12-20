#! /usr/bin/env python

import os, sys, random, re, math, pickle
from utils import *
from os.path import *

sys.path.append('../thirdparty/libsvm/python')
from svmutil import *

#feature_list = ['topic', 'str', 'ugr']
feature_list = ['ugr', 'str', 'inquirer', 'liwc']

feature_name = '_'.join(feature_list)

fold_path    = '../cache/folds/'
model_path   = '../cache/models/'
result_path  = '../cache/results/'
develop_path  = '../cache/develop/'
feature_path = '../cache/features/'
score_path   = '../dataset/labels/'

score_type = 'xofy'
overwrite_model = True
asp_dict = {'serv':'service', 'func':'functionality', 'appe':'appearance', 'o':'other',
        'qual':'quality', 'use':'usability', 'price':'price', 'brand':'brand'}

aspect = asp_dict[sys.argv[1]]
genres = sys.argv[2:]

feature_dims = {'ugr':30000, 'topic':100, 'str':10, 'inquirer':100, 'liwc':100}

for genre in genres:
    print 'processing {0}'.format(genre)
    fname_fold = join(fold_path, '{0}.reviews'.format(genre))
    fname_score = join(score_path, '{0}.{1}'.format(genre, score_type))

    folds = [ int(f) for f in get_content(fname_fold) ]
    scores = read_scores_from_file(fname_score)

    num = len(folds)
    features = [ {} for i in range(num) ]
    feat_idx_base = 0
    for feature in feature_list:
        print 'read feature: ', feature
        fname_feature = join(feature_path, '{0}/{1}.feat'.format(genre, feature))
        iter_features = read_features_from_file(fname_feature)

        for i in xrange(num):
            X = iter_features[i]
            feat_idxs = [ idx + feat_idx_base  for idx in X.keys() ]
            feat_values = X.values()
            X = dict(zip(feat_idxs, feat_values))
            features[i] = dict( features[i].items() + X.items() )
        feat_idx_base += feature_dims[feature]

    # Filter out invalid items
    folds, features, scores = filter_out_invalid_items(folds, features, scores)

    for fold_id in fold_ids:
        ###################################################################
        # partition
        y_train, x_train = get_train_data(scores, features, folds, fold_id)
        y_test, x_test = get_test_data(scores, features, folds, fold_id)
        y_dev, x_dev = get_dev_data(scores, features, folds, fold_id)

        print "Processing folder: ", fold_id
        print "Before Sampling: ", len(y_train), len(y_test)
        ###################################################################
        # classification: training and testing
        # Down sampling for genres has too much examples
        fname_asps = join(score_path, '{0}.asps'.format(genre))
        asps = read_asps_from_file(fname_asps)
        dummy_ys = [0] * len(asps)
        _, asps_train = get_train_data(dummy_ys, asps, folds, fold_id)
        _, asps_dev = get_train_data(dummy_ys, asps, folds, fold_id)

        y_train, x_train = filter_by_asps(y_train, x_train, aspect, asps_train)
        x_train, y_train = sample_dataset(x_train, y_train, 10000)
        print "After Sampling: ", len(y_train), len(y_test)

        model_name = '{0}/{1}.{2}.fold_{3}.{4}.libsvm.regression.model'.format(model_path, genre, aspect, fold_id, feature_name)

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
            m = svm_load_model( model_name )

        # test set
        y_pred, mse_test, y_pred_2 = svm_predict(y_test, x_test, m)
        out_file = '{0}/{1}.{2}.fold_{3}.{4}.libsvm.regression.pred'.format(result_path, genre, aspect, fold_id, feature_name)
        fout = open(out_file, 'w+')
        for i in  range(len(y_test)):
            ostr = "{0} {1} {2}\n".format(y_test[i], y_pred[i], y_pred_2[i][0])
            fout.write(ostr)
        fout.close()

        # develop set
        y_pred, mse_test, y_pred_2 = svm_predict(y_dev, x_dev, m)
        out_file = '{0}/{1}.{2}.fold_{3}.{4}.libsvm.regression.pred'.format(develop_path, genre, aspect, fold_id, feature_name)
        fout = open(out_file, 'w+')
        for i in  range(len(y_dev)):
            ostr = "{0} {1} {2}\n".format(y_dev[i], y_pred[i], y_pred_2[i][0])
            fout.write(ostr)
        fout.close()

#! /usr/bin/env python

import os, sys, random, re, math
from utils import *
from os.path import *

sys.path.append('../thirdparty/liblinear/python')
sys.path.append('../thirdparty/libsvm/python')

LIBLINEAR = False

if LIBLINEAR:
    from liblinearutil import *
else:
    from svmutil import *

feature_list = ['ugr', 'str', 'inquirer', 'liwc']

feature_name = '_'.join(feature_list)

data_path = '../dataset_v2/t0/{0}'
cache_path = '../dataset_v2/t0/cache/{0}'

genres = sys.argv[1:]

feature_dims = {'ugr':30000, 'str':10, 'inquirer':100, 'liwc':100}

for genre in genres:
    data_path_genre = data_path.format( genre )
    cache_path_genre = cache_path.format( genre )

    model_path = cache_path_genre + '/models/'
    result_path = cache_path_genre + '/results/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print 'processing {0}'.format(genre)
    fname_fold = join(data_path_genre, 'folds.txt')
    fname_score = join(data_path_genre, 'labels.txt')

    folds = [ int(f) for f in get_content(fname_fold) ]
    #scores = read_scores_from_file_2(fname_score)
    scores = read_scores_from_file(fname_score)

    num = len(folds)
    features = [ {} for i in range(num) ]
    feat_idx_base = 0
    for feature in feature_list:
        print 'read feature: ', feature
        fname_feature = join(cache_path_genre, 'features/{0}.feat'.format(feature))
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

        x_train.extend(x_dev)
        y_train.extend(y_dev)
        print "Processing folder: ", fold_id
        print "Before Sampling: ", len(y_train), len(y_test)
        ###################################################################
        # classification: training and testing
        # Down sampling for genres has too much examples
        x_train, y_train = sample_dataset(x_train, y_train, 20000)
        print "After Sampling: ", len(y_train), len(y_test)

        if False:
            mse = [0]*4 # mseuracy
            for cc in range(4):
                tmpC = math.pow(10, cc-2)
                print 'testing C = ', tmpC
                if LIBLINEAR:
                    mse[cc] = train(y_train, x_train, '-s 11 -v 5 -q -c {0}'.format(tmpC) )
                else:
                    mse[cc] = svm_train(y_train, x_train, '-s 3 -t 2 -v 5 -q -c {0}'.format(tmpC) )
            mse_i = mse.index(min(mse))
            best_c = math.pow(10, mse_i-2)
        else:
            best_c = 1.0
        print 'The best C is: ', best_c

        if LIBLINEAR:
            m = train(y_train, x_train, '-s 11 -q -c {0}'.format(best_c) )
            y_pred, mse_test, y_pred_2 = predict(y_test, x_test, m)

            model_name = '{0}/fold_{1}.{2}.liblinear.regression.model'.format(model_path, fold_id, feature_name);
            save_model( model, m )
            out_file = '{0}/fold_{1}.{2}.liblinear.regression.pred'.format(result_path, fold_id, feature_name)
        else:
            m = svm_train(y_train, x_train, '-s 3 -t 2 -q -c {0}'.format(best_c) )
            y_pred, mse_test, y_pred_2 = svm_predict(y_test, x_test, m)

            model_name = '{0}/fold_{1}.{2}.libsvm.regression.model'.format(model_path, fold_id, feature_name);
            svm_save_model( model_name, m)
            out_file = '{0}/fold_{1}.{2}.libsvm.regression.pred'.format(result_path, fold_id, feature_name)

        fout = open(out_file, 'w+')
        for i in  range(len(y_test)):
            ostr = "{0} {1} {2}\n".format(y_test[i], y_pred[i], y_pred_2[i][0])
            fout.write(ostr)
        fout.close()

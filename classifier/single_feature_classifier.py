#! /usr/bin/env python

import os, sys, random,  math
from utils import *
from os.path import *

sys.path.append('../thirdparty/liblinear/python')
sys.path.append('../thirdparty/libsvm/python')

LIBLINEAR = False

if LIBLINEAR:
    from liblinearutil import *
else:
    from svmutil import *

feature_name = sys.argv[1]

fold_path    = '../cache/folds/'
model_path   = '../cache/models/'
result_path  = '../cache/results/'
feature_path = '../cache/features/'
score_path   = '../dataset/labels/'

score_type = 'xofy'

genres = sys.argv[2:]

for genre in genres:
    print 'processing {0}'.format(genre)
    fname_fold = join(fold_path, '{0}.reviews'.format(genre))
    fname_score = join(score_path, '{0}.{1}'.format(genre, score_type))
    fname_feature = join(feature_path, '{0}/{1}.feat'.format(genre, feature_name))

    folds = [ int(f) for f in get_content(fname_fold) ]
    scores = read_scores_from_file(fname_score)
    features = read_features_from_file(fname_feature)

    # Filter out invalid items
    folds, features, scores = filter_out_invalid_items(folds, features, scores)
    fold_ids = range(10)
    for fold_id in fold_ids:
        ###################################################################
        # partition
        y_train, x_train = get_train_data(scores, features, folds, fold_id)
        y_test, x_test = get_test_data(scores, features, folds, fold_id)
        y_dev, x_dev = get_dev_data(scores, features, folds, fold_id)

        print "Processing folder: ", fold_id
        print "Before Sampling: ", len(y_train), len(y_test)
        #
        #for jj in range(10):
        #    print '#'*50
        #    print y_train[jj]
        #    print x_train[jj]
        #    print '#'*50
        #sys.exit(1)
        ###################################################################
        # classification: training and testing
        # Down sampling for genres has too much examples
        step_size = 1
        if genre in ['outdoor']:
            step_size = 3
        elif genre in ['electronics']:
            step_size = 10
        elif genre in ['home']:
            step_size = 10
        x_train = x_train[1:-1:step_size]
        y_train = y_train[1:-1:step_size]

        if True:
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
            save_model( '{0}/{1}.{2}.fold_{3}.{4}.liblinear.regression.model'.format(model_path, genre, score_type, fold_id, feature_name), m );
            out_file = '{0}/{1}.{2}.fold_{3}.{4}.liblinear.regression.pred'.format(result_path, genre, score_type, fold_id, feature_name)
        else:
            m = svm_train(y_train, x_train, '-s 3 -t 2 -q -c {0}'.format(best_c) )
            #print "Training error: "
            #y_pred, mse_test, y_pred_2 = svm_predict(y_train, x_train, m)
            #print "Testing error: "
            y_pred, mse_test, y_pred_2 = svm_predict(y_test, x_test, m)
            svm_save_model( '{0}/{1}.{2}.fold_{3}.{4}.libsvm.regression.model'.format(model_path, genre, score_type, fold_id, feature_name), m );
            out_file = '{0}/{1}.{2}.fold_{3}.{4}.libsvm.regression.pred'.format(result_path, genre, score_type, fold_id, feature_name)

        fout = open(out_file, 'w+')
        for i in  range(len(y_test)):
            ostr = "{0} {1} {2}\n".format(y_test[i], y_pred[i], y_pred_2[i][0])
            fout.write(ostr)
        fout.close()

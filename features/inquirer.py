#! /usr/bin/env python

import os, sys, random, yaml, re
sys.path.append('/Users/yinfei.yang/workspace/nlp/amazon_review/helpfulness/src/helpfulness/dict/INQUIRER')
from inquirer_helper import load_inquirer_dict, compute_feature_inquirer_raw

feature_path = '../dataset_v2/t5/cache/{0}/features/'
data_path = '../dataset_v2/t5/{0}/'

genres = sys.argv[1:]

inquirer_dict, inquirer_keys = load_inquirer_dict()

for genre in genres:
    genre_path = data_path.format(genre)
    genre_feature_path = feature_path.format(genre)

    if not os.path.exists(genre_feature_path):
        os.makedirs(genre_feature_path)

    fname_feature = os.path.join(*[genre_feature_path, 'inquirer.feat'])
    fout = open(fname_feature, 'w+')

    fname_review = os.path.join(genre_path, 'reviews.txt')
    with open(fname_review) as fin:
        reviews = fin.readlines()
        for review in reviews:
            review = review.replace('|', ' ')
            feat_dict = compute_feature_inquirer_raw(review, inquirer_keys, inquirer_dict)

            str = ''
            for i in range( len(inquirer_keys) ):
                key = inquirer_keys[i]
                idx = i+1

                if feat_dict.has_key(key):
                    feat = feat_dict[key]
                else:
                    feat = 0

                if feat > 0:
                    str += ' {0}:{1}'.format(idx, feat)
            fout.write('{0}\n'.format( str.strip() ) )
    fout.close()

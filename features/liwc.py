#! /usr/bin/env python

import os, sys, random, yaml, re
sys.path.append('/Users/yinfei.yang/workspace/nlp/amazon_review/helpfulness/src/helpfulness/dict/LIWC')
from liwc_helper import load_liwc_dict, compute_feature_liwc_raw

feature_path = '../dataset_v2/t0/cache/{0}/features/'
data_path = '../dataset_v2/t0/{0}/'

genres = sys.argv[1:]

liwc_dict, liwc_keys = load_liwc_dict()

for genre in genres:
    genre_path = data_path.format(genre)
    genre_feature_path = feature_path.format(genre)

    if not os.path.exists(genre_feature_path):
        os.makedirs(genre_feature_path)

    fname_feature = os.path.join(*[genre_feature_path, 'liwc.feat'])
    print 'Storing features at: ', fname_feature
    fout = open(fname_feature, 'w+')

    fname_review = os.path.join(genre_path, 'reviews.txt')
    with open(fname_review) as fin:
        reviews = fin.readlines()
        for review in reviews:
            review = review.replace('|', ' ')
            review = review.lower()
            feat_dict = compute_feature_liwc_raw(review, liwc_keys, liwc_dict)

            str = ''
            for i in range( len(liwc_keys) ):
                key = liwc_keys[i]
                idx = i+1

                if feat_dict.has_key(key):
                    feat = feat_dict[key]
                else:
                    feat = 0

                if feat > 0:
                    str += ' {0}:{1}'.format(idx, feat)
            fout.write('{0}\n'.format( str.strip() ) )
    fout.close()

#! /usr/bin/env python

import os, sys, random, yaml, re
sys.path.append('/Users/yinfei.yang/workspace/nlp/amazon_review/helpfulness/src/helpfulness/dict/LIWC')
from liwc_helper import load_liwc_dict, compute_feature_liwc_raw

feature_path = '../cache/features/'
review_path = '../dataset/reviews_lemma/'

genres = sys.argv[1:]

liwc_dict, liwc_keys = load_liwc_dict()

for genre in genres:
    fname_feature = os.path.join(*[feature_path, genre, 'liwc.feat'])
    fout = open(fname_feature, 'w+')

    fname_review = os.path.join(review_path, '{0}.reviews'.format(genre))
    with open(fname_review) as fin:
        reviews = fin.readlines()
        for review in reviews:
            review = review.replace('|', ' ')
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

#! /usr/bin/env python

import os, sys, math, re

feature_path = '../cache/features/'
dict_path = '../lexical/lexical_dict/'
review_path = '../dataset/reviews_lemma/'

genres = sys.argv[1:]

for genre in genres:
    ugr_dict_file = os.path.join(dict_path, '{0}_cutoff.ugr'.format(genre))
    ugr_dict = {}
    feat_keys = []
    with open(ugr_dict_file, 'r') as f:
        for line in f.readlines():
            ts = line.strip().split()
            word = ts[0]
            df = int(ts[1])

            ugr_dict[word] = df
            feat_keys.append(word)

    fname_feature = os.path.join(*[feature_path, genre, 'ugr.feat'])
    fout = open(fname_feature, 'w+')

    fname_review = os.path.join(review_path, '{0}.reviews'.format(genre))
    with open(fname_review) as fin:
        reviews = fin.readlines()
        N = len(reviews)
        for review in reviews:
            tf_dict = {}
            for key in feat_keys:
                tf_dict[key] = 0

            sents = review.split('|')
            for sent in sents:
                words = sent.split()
                for word in words:
                    if tf_dict.has_key(word):
                            tf_dict[word] += 1

            str = ''
            for i in range( len(feat_keys) ):
                key = feat_keys[i]

                idx = i+1 # feature idx start from 1

                tf = tf_dict[key]
                df = ugr_dict[key]
                feat = 1.0 * tf * math.log10(1.0*N/df)
                if feat > 0:
                    str += ' {0}:{1:.3f}'.format(idx, feat)
            fout.write('{0}\n'.format( str.strip() ) )
    fout.close()

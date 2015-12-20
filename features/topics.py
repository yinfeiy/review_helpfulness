#! /usr/bin/env python

import os, sys

feature_path = '../cache/features/'
data_path = '../dataset/topics/'

genres = sys.argv[1:]

for genre in genres:
    topic_fname = os.path.join(data_path, '{0}.topics'.format(genre))
    fname_feature = os.path.join(*[feature_path, genre, 'topic.feat'])
    fout = open(fname_feature, 'w+')

    lines = open(topic_fname).readlines()

    for line in lines:
        feats = [int(t) for t in line.strip().split()]

        str = ''
        for i in range(len(feats)):
            feat = feats[i]
            idx = i+1

            if feat > 0:
                str += ' {0}:{1}'.format(idx, feat)

        fout.write('{0}\n'.format( str.strip() ) )
    fout.close()

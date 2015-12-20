#! /usr/bin/env python

import os, sys, random, re

feature_path = '../cache/features/'
meta_path = '../dataset/metas/'
pid_path = '../dict/PID/'

genres = sys.argv[1:]

for genre in genres:

    fname_feature = os.path.join(*[feature_path, genre, 'meta.feat'])
    fout = open(fname_feature, 'w+')
    fname_pid = os.path.join(pid_path, '{0}_pid.dcit'.format(genre))
    fout_pid = open(fname_pid, 'w+')

    fname_meta = os.path.join(meta_path, '{0}.meta'.format(genre))
    metas = open(fname_meta).readlines()

    base_id = 1
    pid_dict = {}
    for meta in metas:
        ts = meta.strip().split()
        prod_type = ts[0]
        star = float(ts[1])

        if pid_dict.has_key(prod_type):
            pid = pid_dict[prod_type]
        else:
            pid = base_id
            base_id += 1
            pid_dict[prod_type] = pid
            fout_pid.write('{0} {1}\n'.format(pid, prod_type))

        feature = [pid, star]

        str = ''
        for i in range( len(feature) ):
            idx = i+1 # feature idx start from 1
            feat = feature[i]
            if feat > 0:
                str += ' {0}:{1}'.format(idx, feat)
        fout.write('{0}\n'.format( str.strip() ) )
    fout.close()

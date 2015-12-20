import os, sys
import numpy as np
from scipy.stats import spearmanr

label_path = '../dataset/labels/'
feature_path = '../cache/features/'
twitter_result_path ='/Users/yinfei.yang/workspace/nlp/amazon_review/aaai/data/twitter_lda_results/'

genres = ['electronics', 'cellphone', 'home', 'outdoor', 'watches']

def load_topics(topics_file):
    topics = {}
    with open(topics_file) as fin:
        for line in fin.readlines():
            if line[0] != '\t':
                ts = line.strip().split('\t')
                key = int(ts[0].replace('Topic ','').replace(':','').strip())+1
                value = ts[1]

                topics[key] = [value]
            else:
                ts = line.strip().split('\t')
                value = ts[0]
                topics[key].append(value)
    return topics

def load_labels(labels_file):
    labels = []
    with open(labels_file) as fin:
        for line in fin.readlines():
            ts = line.strip().split('/')
            x = float(ts[0])
            y = float(ts[1])
            score = x/y
            labels.append(score)
    return labels

def load_feates(feates_file):
    feates = []
    with open(feates_file) as fin:
        for line in fin.readlines():
            feat = {}
            ts = line.strip().split()
            for t in ts:
                fs = t.split(':')
                k = int(fs[0])
                v = int(fs[1])
                feat[k] = v
            feates.append(feat)
    return feates

for genre in genres:
    fname_l = os.path.join(label_path, '{0}.xofy'.format(genre))
    fname_f = os.path.join(*[feature_path, genre, 'topic.feat'])
    fname_t = os.path.join(*[twitter_result_path, genre, 'WordsInTopics.txt'])

    topics = load_topics(fname_t)
    labels = load_labels(fname_l)
    feates = load_feates(fname_f)

    num = len(labels)
    corrs = []
    for i in range(1, 101):
        xs = []
        for n in range(num):
            feat = feates[n]
            if feat.has_key(i):
                xs.append(feat[i])
            else:
                xs.append(0)
        corr, p = spearmanr(xs, labels)
        corrs.append(corr)

    idxs = np.argsort(corrs)
    idxs = idxs[::-1]

    fout = open(genre + '.txt', 'w+')
    for idx in idxs:
        if not np.isnan(corrs[idx]):
            fout.write( '{0} {1}:'.format(idx, corrs[idx]))
            ts = topics[idx+1]
            ws = ' '.join(ts[:20])
            fout.write(ws)
            fout.write('\n')
    fout.close()

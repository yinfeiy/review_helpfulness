#! /usr/bin/env python

import os, sys, random, re

feature_path = '../dataset_v2/t0/cache/{0}/features/'
data_path = '../dataset_v2/t0/{0}/'

genres = sys.argv[1:]

for genre in genres:
    genre_path = data_path.format(genre)
    genre_feature_path = feature_path.format(genre)

    if not os.path.exists(genre_feature_path):
        os.makedirs(genre_feature_path)

    fname_feature = os.path.join(*[genre_feature_path, 'str.feat'])
    print 'Storing features at: ', fname_feature

    fout = open(fname_feature, 'w+')
    fname_review = os.path.join(genre_path, 'reviews.txt')

    features = []
    reviews = open(fname_review).readlines()
    for review in reviews:
        sents = review.split('|')
        feature = []
        # total number of tokens
        num_token = 0
        # the number of exclamation marks
        num_exclamation = 0
        # number of sentences
        num_sent = 0
        # number of question sentences
        num_question_sent = 0
        for sent in sents:
            sent = sent.strip()
            if len(sent)==0:
                continue
            num_sent += 1

            tokens = sent.split(' ')
            num_token += len(tokens)
            for token in tokens:
                pattern = re.compile(r'!')
                num_exclamation += len( re.findall(pattern, token) )
            if token[-1] == '?':
                num_question_sent += 1
        # the average sentence length
        avg_sent_leng = round(1.0*num_token / num_sent, 3)

        # the precentage of question sentences
        precent_question_sent = round(1.0*num_question_sent / num_sent, 3)

        feature = [num_token, num_exclamation, num_sent, avg_sent_leng, precent_question_sent]

        str = ''
        for i in range( len(feature) ):
            idx = i+1 # feature idx start from 1
            feat = feature[i]
            if feat > 0:
                str += ' {0}:{1}'.format(idx, feat)
        fout.write('{0}\n'.format( str.strip() ) )
    fout.close()

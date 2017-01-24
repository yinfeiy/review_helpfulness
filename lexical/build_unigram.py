import os, sys, random
from glob import glob
from sets import Set

data_path = '../dataset_v2/t5/{0}/'

genres = sys.argv[1:]

for genre in genres:
    print 'processing ', genre
    fname = os.path.join(data_path.format(genre), 'reviews.txt')

    ofname = os.path.join(data_path.format(genre), 'ugr.dict')
    ugr_df = {}

    print genre
    with open(fname, 'r') as stream:

        reviews = stream.readlines()
        for review in reviews:
            sents = review.split('|')

            for sent in sents:
                words = sent.split()
                words = Set(words)
                for word in words:
                    if ugr_df.has_key(word):
                        ugr_df[word] += 1
                    else:
                        ugr_df[word] = 1

    fout = open(ofname, 'w+')
    for key in ugr_df.keys():
        fout.write('{0} {1}\n'.format(key, ugr_df[key]))
    fout.close()

import os, sys, random
from glob import glob
from sets import Set

in_path = '../dataset/reviews_lemma/'
out_path = './lexical_dict/'

genres = sys.argv[1:]

for genre in genres:
    print 'processing ', genre
    fname = os.path.join(in_path, genre+'.reviews')

    ofname = os.path.join(out_path, genre+'.ugr')
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

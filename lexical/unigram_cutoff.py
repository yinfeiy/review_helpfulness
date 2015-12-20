import os, sys, random, re
from glob import glob
from sets import Set

dict_path = './lexical_dict/'

# Load stop words
stop_file = 'english.stop'
stop_words = Set([])
with open(stop_file, 'r') as f:
    for line in f.readlines():
        stop_words.add(line.strip())


genres = sys.argv[1:]

# Remove low frequet words (<3 ) and stop words
for genre in genres:
    ifname = os.path.join(dict_path, '{0}.ugr'.format(genre))
    ofname = os.path.join(dict_path, '{0}_cutoff.ugr'.format(genre))

    fout = open(ofname, 'w+')
    with open(ifname, 'r') as f:
        for line in f.readlines():
            ts = line.strip().split()
            if len(ts) < 2:
                continue
            word = ts[0]
            df = int(ts[1])

            if df < 20 or word in stop_words:
                continue

            fout.write(line)



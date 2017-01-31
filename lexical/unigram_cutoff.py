import os, sys, random, re
from glob import glob
from sets import Set

data_path = '../dataset_v2/t5/{0}/'
max_num = 30000

# Load stop words
stop_file = 'english.stop'
stop_words = Set([])
with open(stop_file, 'r') as f:
    for line in f.readlines():
        stop_words.add(line.strip())


genres = sys.argv[1:]

# Remove low frequet words (<5 ) and stop words
for genre in genres:
    ifname = os.path.join(data_path.format(genre), 'ugr.dict'.format(genre))
    ofname = os.path.join(data_path.format(genre), 'ugr_cutoff.dict'.format(genre))

    fout = open(ofname, 'w+')
    cnt = 0
    with open(ifname, 'r') as f:
        for line in f.readlines():
            ts = line.strip().split()
            if len(ts) < 2:
                continue
            word = ts[0]
            df = int(ts[1])

            if df < 5 or word in stop_words:
                continue

            fout.write(line)

            cnt+=1
            if cnt >= max_num:
                break



import os
import sys
from sets import Set

data_path = '../inquirerTags.txt'

fin = open(data_path, 'r')
lines = fin.readlines()
fin.close()

tags = Set([])
for line in lines:
    tokens = line.strip().split('\t')
    word = tokens[0]
    for tid in range(1,len(tokens)):
        tags.add(tokens[tid].strip())

for tag in tags:
    print tag

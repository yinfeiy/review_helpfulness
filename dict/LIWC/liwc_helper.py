import os
import sys
import math

LIWC_path = os.path.dirname(os.path.abspath(__file__))
dict_path = os.path.join(LIWC_path, 'LIWC2007_English080730.dic')

LIWC_DIM = 64

class word:
    def __init__(self, w, f):
        self.word = w
        self.feat = f

    def __str__(self):
        ostr = self.word + ': ['
        for f in self.feat:
            ostr = ostr + ' ' + (str)(f) + ' '
        ostr = ostr + ']'
        return ostr

def load_liwc_dict():
    f = open(dict_path, 'r')
    lines = f.readlines()
    f.close()

    liwc_dict = {}
    liwc_keys = []
    flag = False
    # parse classes first
    for line in lines[1:]:
        if line[0] == '%':
            flag = True

        if not flag:
            line = line.strip()
            tokens = line.split('\t')
            liwc_keys.append( (int)(tokens[0]) )
            continue

        f = []
        line = line.strip()
        tokens = line.split('\t')

        w = tokens[0].replace('*', '').lower()
        tokens = tokens[1:]
        for token in tokens:
            f.append((int)(token.strip()))
        liwc_dict[w] = word(w, f)
    return liwc_dict, liwc_keys

def compute_feature_liwc_raw(text, liwc_keys = None, liwc_dict = None):
    if not liwc_dict or not liwc_keys:
        liwc_dict, liwc_keys = load_liwc_dict()

    feat_dict = {}
    for key in liwc_keys:
        feat_dict[key] = 0

    words = text.lower().split()
    for word in words:
        if liwc_dict.has_key(word):
            feats = liwc_dict[word].feat
            for f in feats:
                feat_dict[f] = feat_dict[f] + 1

    return feat_dict


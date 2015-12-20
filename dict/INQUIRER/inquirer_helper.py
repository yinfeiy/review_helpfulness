import os, sys

INQUIRER_path = os.path.dirname(os.path.abspath(__file__))
INQUIRER_DICT_FILE = os.path.join(INQUIRER_path, 'inquirerTags.txt')
INQUIRER_KEYS_FILE = os.path.join(INQUIRER_path, 'tags')

INQUIRER_DIM = 100

class word:
    def __init__(self, w, i, p, po, d):
        self.word = w
        self.intensity = i
        self.pos = p
        self.pol = po
        self.dict = d

    def __str__(self):
        return '(' + self.word + ',' + self.pos + ') ' + self.pol + ' ' + self.intensity + ' from ' + self.dict

    def set_inquirer_value(self, inquirer):
        self.inquirer = inquirer

    def get_inquirer_value(self):
        return self.inquirer

def get_content(fname):
    return open(fname).readlines()

def load_inquirer_dict():
    keys = []
    lines = get_content(INQUIRER_KEYS_FILE)
    for line in lines:
        key = line.strip().lower()
        keys.append(key)
    keys.sort()

    inquirer_dict = {}
    lines = get_content(INQUIRER_DICT_FILE)
    for line in lines:
        f = []
        line = line.strip().lower()
        tokens = line.split('\t')
        wor = tokens[0].replace('*', '')
        tokens = tokens[1:]
        for token in tokens:
            f.append(token.strip())

        nword = word(wor, -1, 'unkonwn', 'unknown', 'INQUIRER')
        nword.set_inquirer_value(f)

        inquirer_dict[wor] = nword

    return inquirer_dict, keys

def compute_feature_inquirer(text, inquirer_keys = None, inquirer_dict = None):
    if not inquirer_dict or not inquirer_keys:
        inquirer_dict, inquirer_keys = load_inquirer_dict()

    feat_dict = {}
    for key in inquirer_keys:
        feat_dict[key] = 0

    words = text.lower().split()

    for word in words:
        if inquirer_dict.has_key(word):
            values = inquirer_dict[word].get_inquirer_value()
            for value in values:
                feat_dict[value] += 1

    feat = []
    for i in range(len(inquirer_keys)):
        key = inquirer_keys[i]
        value = feat_dict[key]

        if value > 0:
            feat.append( dict(zip([i+1], [value])) )

    return feat


def compute_feature_inquirer_raw(text, inquirer_keys = None, inquirer_dict = None):
    if not inquirer_dict or not inquirer_keys:
        inquirer_dict, inquirer_keys = load_inquirer_dict()

    feat_dict = {}
    for key in inquirer_keys:
        feat_dict[key] = 0

    words = text.lower().split()

    for word in words:
        if inquirer_dict.has_key(word):
            values = inquirer_dict[word].get_inquirer_value()
            for value in values:
                feat_dict[value] += 1

    # Normalization
    # N = len(words)
    #for key in inquirer_keys:
    #    feat_dict[key] = feat_dict[key] * 1.0 / N
    #feat_dict[len(inquirer_keys)] = N
    return feat_dict

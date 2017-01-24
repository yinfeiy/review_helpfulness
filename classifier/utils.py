from random import shuffle

fold_ids = range(10)

def get_content(fname):
    fin = open(fname, 'r')
    lines = fin.readlines()
    fin.close()
    return lines

def write_feature_to_file(features, fname):
    raise NotImplementedError()

def read_scores_from_file(fname):
    scores = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            ts = line.strip().split('/')
            x = float(ts[0])
            y = float(ts[1])
            if y >= 1:
                score = x/y
            else:
                score = -1
            scores.append(score)
    return scores

def read_scores_from_file_2(fname):
    scores = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            ts = line.strip().split()
            score = float(ts[0])
            scores.append(score)
    return scores

def read_asps_from_file(fname):
    asps = []
    with open(fname) as f:
        for line in f.readlines():
            asps.append(set(line.strip().split()))
    return asps

def read_features_from_file(fname):
    features = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            feature = {}
            tokens = line.strip().split()
            for token in tokens:
                kv_pair = token.split(':')
                if len(kv_pair) < 2:
                    continue
                key = int(kv_pair[0])
                val = float(kv_pair[1])
                feature[key] = val
            features.append(feature)
    return features

def get_train_data(ys, xs, folds, fold_id):
    fold_ids_rerange = fold_ids[fold_id:]
    fold_ids_rerange.extend(fold_ids[:fold_id])
    train_fold_ids = fold_ids_rerange[1:6]

    train_idx = [idx for idx in range(len(ys)) if folds[idx] in train_fold_ids]
    x_train = [ xs[idx] for idx in train_idx ]
    y_train = [ ys[idx] for idx in train_idx ]

    return y_train, x_train

def get_test_data(ys, xs, folds, fold_id):
    fold_ids_rerange = fold_ids[fold_id:]
    fold_ids_rerange.extend(fold_ids[:fold_id])
    test_fold_id = fold_ids_rerange[0]

    test_idx = [idx for idx in range(len(ys)) if folds[idx] == test_fold_id]
    x_test = [ xs[idx] for idx in test_idx ]
    y_test = [ ys[idx] for idx in test_idx ]

    return y_test, x_test

def get_dev_data(ys, xs, folds, fold_id):
    fold_ids_rerange = fold_ids[fold_id:]
    fold_ids_rerange.extend(fold_ids[:fold_id])
    dev_fold_ids = fold_ids_rerange[6:]

    dev_idx = [idx for idx in range(len(ys)) if folds[idx] in dev_fold_ids]
    x_dev = [ xs[idx] for idx in dev_idx ]
    y_dev = [ ys[idx] for idx in dev_idx ]

    return y_dev, x_dev

def filter_out_invalid_items(folds, features, scores):
    valid_idx = [ idx for idx in range(len(scores)) if scores[idx] >= 0 ]
    folds = [ folds[idx] for idx in valid_idx ]
    features = [ features[idx] for idx in valid_idx ]
    scores = [ scores[idx] for idx in valid_idx ]

    return folds, features, scores

def filter_by_asps(ys, xs, asp, asps):
    valid_idx = [ idx for idx in range(len(asps)) if asp in asps[idx]]
    ys = [ ys[idx] for idx in valid_idx ]
    xs = [ xs[idx] for idx in valid_idx ]

    return ys, xs

def sample_dataset(xs, ys, _size=10000):
    if len(xs) < _size:
        return xs, ys
    idxs = range(len(ys)); shuffle(idxs); idxs = idxs[:_size]
    xs = [ xs[i] for i in idxs]; ys = [ ys[i] for i in idxs]

    return xs, ys

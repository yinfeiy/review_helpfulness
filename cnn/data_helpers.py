import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_and_labels_multi_class(train_data_file, test_data_file, verbose=False):

    # Loda data from files
    train_examples = list(open(train_data_file, "r").readlines())
    train_examples = [te.strip().split('\t' ) for te in train_examples]
    train_text, train_labels_text = [te[0].strip() for te in train_examples], [te[1:] for te in train_examples]

    test_examples = list(open(test_data_file, "r").readlines())
    test_examples = [te.strip().split('\t' ) for te in test_examples]
    test_text, test_labels_text = [te[0].strip() for te in test_examples], [te[1:] for te in test_examples]

    # Split by words
    x_train_text = [clean_str(sent) for sent in train_text]
    x_test_text  = [clean_str(sent) for sent in test_text]

    # Generate labels
    train_labels_count = {}
    for labels_text in train_labels_text:
        for label in labels_text:
            train_labels_count[label] = train_labels_count.get(label, 0) + 1

    labels = list(set([label for labels in train_labels_text for label in labels]))
    labels.sort(key=lambda x:train_labels_count[x], reverse=True)

    if verbose:
        test_labels_count = {}
        for labels_text in test_labels_text:
            for label in labels_text:
                test_labels_count[label] = test_labels_count.get(label, 0) + 1
        for label in labels:
            print label, train_labels_count[label], test_labels_count.get(label, 0)

    y_train = [ [1 if label in labels_text else 0 for label in labels] for labels_text in train_labels_text]
    y_test = [ [1 if label in labels_text else 0 for label in labels] for labels_text in test_labels_text]

    return x_train_text, y_train, x_test_text, y_test, labels


def load_data_and_term_labels(train_data_file, test_data_file):
    # Loda data from files
    train_examples = list(open(train_data_file, "r").readlines())
    train_examples = [te.strip().split('\t' ) for te in train_examples]
    train_text, train_labels_text = [te[0].strip() for te in train_examples], [te[1:] for te in train_examples]

    test_examples = list(open(test_data_file, "r").readlines())
    test_examples = [te.strip().split('\t' ) for te in test_examples]
    test_text, test_labels_text = [te[0].strip() for te in test_examples], [te[1:] for te in test_examples]

    # Split by words
    x_train_text = [clean_str(sent) for sent in train_text]
    x_test_text  = [clean_str(sent) for sent in test_text]

    # Generate labels
    y_train_labels = []
    y_test_labels = []

    labels = set()
    for labels_text in train_labels_text:
        labels_single = []
        for label in labels_text:
            label, term = label.split('|')
            labels.add(label)

            labels_single.append((label, term))
        y_train_labels.append(labels_single)

    for labels_text in test_labels_text:
        labels_single = []
        for label in labels_text:
            label, term = label.split('|')
            labels_single.append((label, term))
        y_test_labels.append(labels_single)

    labels = list(labels)
    labels.sort()
    return x_train_text, y_train_labels, x_test_text, y_test_labels, labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    #train_data_file = '../../data/reviews/review_16_laptop.train'
    #test_data_file = '../../data/reviews/review_16_laptop.test'
    train_data_file = '../../data/reviews/review_16_restaurants_with_term.train'
    test_data_file = '../../data/reviews/review_16_restaurants_with_term.test'

    x_text_train, y_train, x_text_test, y_test, labels = load_data_and_term_labels(train_data_file, test_data_file)

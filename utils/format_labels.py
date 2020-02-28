import numpy as np


def sort(labels, sep=','):
    sorted_labels = sorted(str(labels).split(sep))
    return [label.strip() for label in sorted_labels]


def join(labels):
    return ' '.join(labels)


def get_unique(labels):
    all_labels = sum(labels, [])
    all_labels.append('nan')
    return sorted(list(set(all_labels)))


def encode_onehot(label_list, unique_labels):
    labels = [0] * len(unique_labels)
    indices = [unique_labels.index(label) for label in label_list]
    for ind in indices:
        labels[ind] = 1
    return labels


def decode_onehot(encoding, unique_labels):
    indices = np.where(np.array(encoding) == 1)[0].tolist()
    labels = [unique_labels[ind] for ind in indices]
    return labels

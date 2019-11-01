def sort_labels(labels, sep=','):
    sorted_labels = sorted(str(labels).split(sep))
    return [label.strip() for label in sorted_labels]


def join_labels(labels):
    return ' '.join(labels)


def get_unique_labels(labels):
    all_labels = sum(labels, [])
    all_labels.append('nan')
    return sorted(list(set(all_labels)))


def onehot_encoding(label_list, unique_labels):
    labels = [0] * len(unique_labels)
    indices = [unique_labels.index(label) for label in label_list]
    for ind in indices:
        labels[ind] = 1
    return labels

from collections import Counter
from typing import List

from utils.visualize import bar_plot


def category_distribution(data, col='label'):
    all_assigned_labels: List[str] = sum(data[col].tolist(), [])
    return Counter(all_assigned_labels)


def plot_category_distribution(data, title='', log=False):
    distribution = category_distribution(data)
    keys, frequencies = zip(*sorted(distribution.items()))
    bar_plot(keys, frequencies, title=title, log=log)

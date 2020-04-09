from collections import Counter

from utils.visualize import bar_plot


def category_distribution(data, col='label'):
    counter: Counter = Counter()

    for each_label_list in data[col]:
        counter.update(each_label_list)

    return counter


def plot_category_distribution(data, title='', log=False):
    distribution = category_distribution(data)
    keys, frequencies = zip(*sorted(distribution.items()))
    bar_plot(keys, frequencies, title=title, log=log)

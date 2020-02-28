from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nlkt.metrics import BigramAssocMeasures, TrigramAssocMeasures
from collections import Counter

from utils.visualize import bar_plot


def category_distribution(data, col='label'):
    all_assigned_labels = sum(data[col].tolist(), [])
    return Counter(all_assigned_labels)


def plot_category_distribution(data, title='', log=False):
    distribution = category_distribution(data)
    keys, frequencies = zip(*sorted(distribution.items()))
    bar_plot(keys, frequencies, title=title, log=log)

def collocations(data, col='text', n_gram='bigram'):
    fulltext = ' '.join(data[col].tolist()).lower()
    tokens = fulltext.split()

    if n_gram == 'bigram':
        collocation = BigramCollocationFinder.from_words(tokens)
        n_grams = collocation.nbest(BigramAssocMeasures.likelihood_ratio, 10)
    elif n_gram == 'trigram':
        collocation = TrigramCollocationFinder.from_words(tokens)
        n_grams = collocation.nbest(TrigramAssocMeasures.likelihood_ratio, 10)


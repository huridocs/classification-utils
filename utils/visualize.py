import matplotlib.pyplot as plt


def bar_plot(label, frequencies, log=False, title=''):
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.bar(label, frequencies, log=log)
    plt.xticks(range(len(label)), rotation='vertical')
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()


from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd
import numpy as np


def confusion_matrix(data):
    target = np.array(data.one_hot_labels.tolist())
    prediction = np.array(data.pred_one_hot_label.tolist())
    return multilabel_confusion_matrix(target, prediction)


def metrics(confusion_matrices, categories):
    evaluation = {}
    for ind, matrix in enumerate(confusion_matrices):
        evaluation[categories[ind]] = metrics_from_confusion_matrix(matrix)
    evaluation_df = pd.DataFrame.from_dict(evaluation).transpose()
    evaluation_df.sort_values(['f1', 'pred', 'recall'], ascending=False, inplace=True)
    evaluation_df.loc['mean'] = evaluation_df.mean(axis=0)
    return evaluation_df


def metrics_from_confusion_matrix(matrix):
    tn, fp, fn, tp = matrix.ravel()
    precision = tp / (tp + fn)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return {'f1': round(f1, 4), 'prec': round(precision, 4), 'recall': round(recall, 4)}


def evaluate(data):
    confusion_matrices = confusion_matrix(data)
    evaluation = metrics(confusion_matrices)
    return evaluation


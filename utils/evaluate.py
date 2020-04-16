from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd
import numpy as np


def confusion_matrix(data):
    target = np.array(data.one_hot_labels.tolist())
    prediction = np.array(data.pred_one_hot_label.tolist())
    return multilabel_confusion_matrix(target, prediction)


def compute_metrics(confusion_matrices, categories):
    evaluation = {}
    for ind, matrix in enumerate(confusion_matrices):
        evaluation[categories[ind]] = metrics_from_confusion_matrix(matrix)
    return pd.DataFrame.from_dict(evaluation).transpose()


def metrics_from_confusion_matrix(matrix):
    tn, fp, fn, tp = matrix.ravel()
    precision, recall, f1 = prec_rec_fscore(tp, fp, fn)
    return {
        'f1': round(f1, 4),
        'prec': round(precision, 4),
        'recall': round(recall, 4),
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def evaluate(data, labels):
    confusion_matrices = confusion_matrix(data)
    evaluation = compute_metrics(confusion_matrices, labels)
    micro_avg = micro_average(evaluation)
    evaluation = evaluation.append(pd.DataFrame(micro_avg, index=['micro_avg (weighted average)']))
    evaluation.loc['macro_avg (all labels weigh the same)'] = evaluation.mean(axis=0)
    return evaluation.reindex(
        columns=['f1', 'prec', 'recall', 'tp', 'fp', 'fn'])


def f_score(precision, recall, b=1):
    prec = np.array(precision)
    rec = np.array(recall)
    return (1 + b * b) * (prec * rec) / ((b * b * prec) + rec)


def prec_rec_fscore(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = f_score(precision, recall)
    return precision, recall, f1


def micro_average(eval_df):
    tp = eval_df['tp'].sum()
    fp = eval_df['fp'].sum()
    fn = eval_df['fn'].sum()
    prec, recall, f1 = prec_rec_fscore(tp, fp, fn)
    return {
        'f1': round(f1, 4),
        'prec': round(prec, 4),
        'recall': round(recall, 4)
    }

import tensorflow as tf
import pandas as pd
import pickle
import yaml


def load_csv(path, delimiter=','):
    return pd.read_csv(path, delimiter=delimiter)


def load_yml(path, data_id):
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.BaseLoader)
        return cfg[data_id]


def to_pickle(df, path):
    with tf.gfile.Open(path, 'wb') as f:
        pickle.dump(df, f)


def load_pickle(path):
    with tf.gfile.Open(path, 'rb') as f:
        return pickle.load(f)

import tensorflow as tf
import pandas as pd
import yaml
import gcsfs
import pdb


def load_csv(path, delimiter=','):
    return pd.read_csv(path, delimiter=delimiter)


def load_yml(path, data_id):
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.BaseLoader)
    return cfg[data_id]


def load_from_bucket(bucket_path, delimiter):
    with tf.gfile.Open(bucket_path, 'r') as f:
        data = pd.read_csv(f, delimiter=delimiter)
    return data


def to_pickle(df, path):
    df.to_pickle(path)


def load_pickle(path):
    return pd.read_pickle(path)

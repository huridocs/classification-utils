import tensorflow as tf
import pandas as pd
import pickle
import yaml
from google.cloud import storage
import os


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
    bucket_name = path.split('/')[2]
    file_path = '/'.join(path.split('/')[3:])
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.download_to_filename('download.pkl')

    with open('download.pkl', 'rb') as f:
        data = pickle.load(f)

    os.remove('download.pkl')
    return data

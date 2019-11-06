import prepare
from utils import format_labels, io


def load_data(DATA_ID):
    cfg_path = './config.yml'
    cfg = io.load_yml(cfg_path, DATA_ID)

    try:
        data = io.load_pickle(cfg['pkl_file'])
    except:
        prepare.prepare(DATA_ID, cfg['pkl_file'])
        data = io.load_pickle(cfg['pkl_file'])

    return data


def load_unique_labels(data):
    return format_labels.get_unique(data.label.tolist())






import prepare
from utils import format_labels, io


def load_data(configuration_path, DATA_ID):
    configuration = io.load_yml(configuration_path, DATA_ID)
    try:
        data = io.load_pickle(configuration['pkl_file'])
    except:
        prepare.prepare(DATA_ID, configuration['pkl_file'])
        # data = io.load_pickle(configuration['pkl_file'])
        data = None

    return data


def load_unique_labels(data):
    return format_labels.get_unique(data.label.tolist())

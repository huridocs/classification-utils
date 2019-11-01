from utils import io
from utils import format
import sys


def prepare(data_id, cfg_path='./config.yml'):

    cfg = io.load_yml(cfg_path, data_id)
    data = io.load_csv(cfg['data_path'], cfg['delimiter'])

    data.rename(columns={cfg['text_col']: 'text'}, inplace=True)
    data = data[['text', cfg['label_col']]]
    data.drop_duplicates(inplace=True)

    data['seq_length'] = data.text.map(str.split).apply(len)

    data['label'] = data[cfg['label_col']].apply(format.sort_labels, args=[cfg['sep']])
    data['str_label'] = data['label'].apply(format.join_labels)

    unique_labels = format.get_unique_labels(data.label.tolist())
    data['one_hot_labels'] = data['label'].apply(format.onehot_encoding,
                                                 args=[unique_labels])
    io.to_pickle(data, cfg['output_file'])


if __name__ == '__main__':
    DATA_ID = sys.argv[1]
    if len(sys.argv) > 2:
        cfg_path = sys.argv[2]
        prepare(DATA_ID, cfg_path)
    prepare(DATA_ID)

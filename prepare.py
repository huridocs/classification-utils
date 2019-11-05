from utils import format_labels
from utils import io
import sys
import pdb


def prepare(data_id, cfg_path='./config.yml'):

    cfg = io.load_yml(cfg_path, data_id)
    data = io.load_csv(cfg['data_file'])

    data.rename(columns={cfg['text_col']: 'text'}, inplace=True)
    data = data[['text', cfg['label_col']]]
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

    data['seq_length'] = data.text.map(str.split).apply(len)

    data['label'] = data[cfg['label_col']].apply(format_labels.sort,
                                                 args=[cfg['sep']])
    data['str_label'] = data['label'].apply(format_labels.join)

    unique_labels = format_labels.get_unique(data.label.tolist())
    data['one_hot_labels'] = data['label'].apply(format_labels.encode_onehot,
                                                 args=[unique_labels])
    io.to_pickle(data, cfg['pkl_file'])


if __name__ == '__main__':
    DATA_ID = sys.argv[1]
    if len(sys.argv) > 2:
        cfg_path = sys.argv[2]
        prepare(DATA_ID, cfg_path)
    else:
        prepare(DATA_ID)

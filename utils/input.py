from . import tokenization


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def get_label(row, label_col, set_type):
    if set_type == 'test':
        return 'nan'
    return tokenization.convert_to_unicode(row[label_col])


def get_multilabels(row, label_col, nr_labels, set_type):
    if set_type == 'test':
        return [0] * nr_labels
    return row[label_col]


def create_examples(data, text_col, label_col, nr_labels=None,
                    set_type='train', mode='multilabel'):
    examples = []
    for index, row in data.iterrows():
        guid = "%s-%s" % (set_type, index)
        text_a = tokenization.convert_to_unicode(row[text_col])
        if mode == 'singlelabel':
            label = get_label(row, label_col, set_type)
        if mode == 'multilabel':
            label = get_multilabels(row, label_col, nr_labels, set_type)
        examples.append(InputExample(guid=guid, text_a=text_a, label=label))
    return examples


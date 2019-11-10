from utils.format_labels import decode_onehot
import pandas as pd


def get_prediction(probabilities, threshold):
    result = [1 if probability >= threshold else 0 for probability in probabilities]
    if sum(result) == 0:
        result = [0] * len(probabilities)
        result[probabilities.argmax()] = 1
    return result


def format_predictions(data, results, labels, threshold=0.5):
    data = data[:len(results)]
    data['pred_one_hot_label'] = [get_prediction(probabilities, threshold) for
                                  probabilities in results]
    data['pred_label'] = data['pred_one_hot_label'].apply(decode_onehot, args=[labels])
    data['pred_prob_all'] = [list(probabilities) for probabilities in results]
    data['pred_prob'] = data.apply(lambda x: sorted(x.pred_prob_all, reverse=True)[:len(x.pred_label)], axis=1)
    return data


def sample(data, max_T=2, max_F=4):
    Ts = data[data['pred_res'] == 'T']
    Fs = data[data['pred_fes'] == 'F']
    if len(Ts) > max_T:
        Ts = Ts.sample(max_T)
    if len(Fs) > max_F:
        Fs = Fs.sample(max_F)
    return pd.concat([Ts, Fs])

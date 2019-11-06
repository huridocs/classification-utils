import pandas as pd


def sample_data(data, labels, MAX_N):
    samples = []
    for label in labels:
        category_data = data[data.str_label.str.contains(label)]
        if len(category_data) < MAX_N:
            samples.append(category_data.sample(len(category_data)))
        else:
            samples.append(category_data.sample(MAX_N))
    samples_df = pd.concat(samples)
    samples_df.drop_duplicates(subset=['text', 'str_label'], inplace=True)
    return samples_df

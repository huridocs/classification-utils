import pandas as pd


def sample_data(data, labels, MAX_N, random_state=None):
    samples = []
    for ind, label in enumerate(labels):
        category_data = data[data.str_label.str.contains(label)]
        if len(category_data) < MAX_N:
            samples.append(category_data.sample(len(category_data)),
                           random_state=random_state)
        else:
            samples.append(category_data.sample(MAX_N),
                           random_state=random_state)
    samples_df = pd.concat(samples)
    samples_df.drop_duplicates(subset=['text', 'str_label'], inplace=True)
    return samples_df

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def pre_process(s: str):
    return ''.join(ch.lower() for ch in ' '.join(s.split()) if ch.isalnum() or ch == ' ')


if __name__ == '__main__':
    hackathon_order = pd.read_csv('Doc.ua/hackathon_order.csv', error_bad_lines=False)
    doc_spec = pd.read_csv('Doc.ua/doc_spec.csv', error_bad_lines=False)
    # speciality = pd.read_csv('Doc.ua/specialty.csv', error_bad_lines=False)

    hackathon_order['doctor_id'] = pd.to_numeric(hackathon_order['doctor_id'], errors='coerce')
    hackathon_order.dropna(inplace=True, subset=['doctor_id'])
    hackathon_order['doctor_id'] = hackathon_order['doctor_id'].astype(np.int64)

    hackathon_order['comment'] = hackathon_order['comment'].apply(lambda s: pre_process(s) if isinstance(s, str) else s)
    hackathon_order['comment'].replace('', np.nan, inplace=True)
    hackathon_order.dropna(inplace=True, subset=['comment'])

    completed_orders = hackathon_order[hackathon_order['mainStatus'] == 'complete']
    comment_specialty = \
        hackathon_order[['comment', 'doctor_id']].merge(  # !!! hackathon_order completed_orders
            doc_spec[['specialty_id', 'doctor_id']],
            on='doctor_id',
        ).drop('doctor_id', axis=1)


    def plot_dist(data, threshold):
        prob = data.value_counts(normalize=True)
        mask = prob > threshold
        tail_prob = prob.loc[~mask].sum()
        prob = prob.loc[mask]
        prob['other'] = tail_prob
        prob.plot(kind='bar')
        plt.xticks(rotation=25)


    plot_dist(comment_specialty['specialty_id'], threshold=0.0278)
    plt.figure()
    plot_dist(comment_specialty['specialty_id'], threshold=0.)
    plt.show()

    spec_dist = comment_specialty['specialty_id'].value_counts(normalize=True)
    popular_specs = spec_dist[spec_dist > 0.0278].index.to_list()  # [3, 13, 15, 52, 61, 56, 8, 23, 53, 28, 49]

    comment_specialty = comment_specialty[comment_specialty['specialty_id'].isin(popular_specs)]
    comment_specialty = comment_specialty.groupby('comment')['specialty_id'].apply(list).reset_index()

    mlb = MultiLabelBinarizer(sparse_output=False, classes=popular_specs)
    comment_specialty['classes'] = mlb.fit_transform(comment_specialty['specialty_id']).tolist()
    comment_specialty.drop('specialty_id', axis=1, inplace=True)
    comment_specialty.to_csv('comment_classes.csv')

    pass

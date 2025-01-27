import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def pre_process(s: str):
    for c in [':', '-', '\n']:
        s = s.replace(c, ' ')
    lower_alpha_space = ''.join(ch for ch in s if ch.isalnum() or ch in [' ', ',', '.']).lower()
    one_space = ' '.join(lower_alpha_space.split())
    return one_space


if __name__ == '__main__':
    hackathon_order = pd.read_csv('Doc.ua/hackathon_order.csv', error_bad_lines=False, escapechar='\\')
    doc_spec = pd.read_csv('Doc.ua/doc_spec.csv', escapechar='\\')
    speciality = pd.read_csv('Doc.ua/specialty.csv', escapechar='\\')
    disease = pd.read_csv('Doc.ua/disease.csv', escapechar='\\')
    disease_symptom = pd.read_csv('Doc.ua/disease_symptom.csv', escapechar='\\')
    symptom = pd.read_csv('Doc.ua/symptom.csv', escapechar='\\')

    hackathon_order['doctor_id'] = pd.to_numeric(hackathon_order['doctor_id'], errors='coerce')
    hackathon_order.dropna(inplace=True, subset=['doctor_id'])
    hackathon_order['doctor_id'] = hackathon_order['doctor_id'].astype(np.int64)

    hackathon_order['comment'] = hackathon_order['comment'].apply(
        lambda s: pre_process(s) if isinstance(s, str) else np.nan
    )
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


    plot_dist(comment_specialty['specialty_id'], threshold=0.012)
    plt.figure()
    plot_dist(comment_specialty['specialty_id'], threshold=0.)
    plt.show()

    spec_dist = comment_specialty['specialty_id'].value_counts(normalize=True)
    popular_specs = spec_dist[spec_dist > 0.012].index.to_list()

    comment_specialty = comment_specialty[comment_specialty['specialty_id'].isin(popular_specs)]
    comment_specialty = comment_specialty.groupby('comment')['specialty_id'].apply(list).reset_index()

    mlb = MultiLabelBinarizer(sparse_output=False, classes=popular_specs)
    comment_specialty['labels'] = mlb.fit_transform(comment_specialty['specialty_id']).tolist()
    comment_specialty.drop('specialty_id', axis=1, inplace=True)
    comment_specialty.rename(columns={'comment': 'text'}, inplace=True)
    comment_specialty.to_csv('data/comment_specialty.csv', index=False)

    keys, counts = np.unique(comment_specialty.text.map(len).to_list(), return_counts=True)
    plt.bar(keys, counts)
    plt.show()

    counts = np.cumsum(counts)
    counts = counts / counts[-1]
    plt.axhline(0.95)
    plt.bar(keys, counts)
    plt.show()

    speciality_id_name = speciality[['id', 'name']].set_index('id').loc[popular_specs].reset_index()
    speciality_id_name.to_csv('data/speciality_id_name.csv', index=False)

    disease['name'] = disease['name'].apply(lambda s: pre_process(s) if isinstance(s, str) else np.nan)
    disease.dropna(inplace=True, subset=['name'])
    comment_disease = disease[['id', 'name']] \
        .merge(disease_symptom, left_on='id', right_on='disease_id') \
        .merge(symptom[['id', 'name']], left_on='symptom_id', right_on='id') \
        .groupby('name_x')['name_y'].apply(lambda s: pre_process(' '.join(s))).reset_index()

    # comment_disease['name_x'] = np.identity(len(comment_disease), dtype=int).tolist()
    comment_disease.rename(columns={'name_x': 'disease', 'name_y': 'symptoms'}, inplace=True)
    comment_disease.to_csv('data/disease_symptoms.csv', index=False)

    pass

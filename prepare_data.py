# Copyright 2021 by Dmytro Vynnyk.
# All rights reserved.
# This file is part of the your-doc-bot Telegram bot,
# and is released under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/
# or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.".
# Please see the LICENSE file that should have been included as part of this package.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def pre_process(s: str):
    return ''.join(ch.lower() for ch in ' '.join(s.split()) if ch.isalnum() or ch == ' ')


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


    plot_dist(comment_specialty['specialty_id'], threshold=0.015)
    plt.figure()
    plot_dist(comment_specialty['specialty_id'], threshold=0.)
    # plt.show()

    spec_dist = comment_specialty['specialty_id'].value_counts(normalize=True)
    popular_specs = spec_dist[spec_dist > 0.012].index.to_list()  # [3, 13, 15, 52, 61, 56, 8, 23, 53, 28, 49]

    comment_specialty = comment_specialty[comment_specialty['specialty_id'].isin(popular_specs)]
    comment_specialty = comment_specialty.groupby('comment')['specialty_id'].apply(list).reset_index()

    mlb = MultiLabelBinarizer(sparse_output=False, classes=popular_specs)
    comment_specialty['classes'] = mlb.fit_transform(comment_specialty['specialty_id']).tolist()
    comment_specialty.drop('specialty_id', axis=1, inplace=True)
    comment_specialty.to_csv('data/comment_specialty.csv', index=False)

    speciality_id_name = speciality[['id', 'name']].set_index('id').loc[popular_specs].reset_index()
    speciality_id_name.to_csv('data/speciality_id_name.csv', index=False)

    disease['name'] = disease['name'].apply(lambda s: pre_process(s) if isinstance(s, str) else np.nan)
    disease.dropna(inplace=True, subset=['name'])
    dis_name_descript = disease[['id', 'name']] \
        .merge(disease_symptom, left_on='id', right_on='disease_id') \
        .merge(symptom[['id', 'name']], left_on='symptom_id', right_on='id')\
        .groupby('name_x')['name_y'].apply(lambda s: pre_process(' '.join(s))).reset_index()

    # dis_name_descript['name_x'] = np.identity(len(dis_name_descript), dtype=int).tolist()
    comment_disease = pd.DataFrame({'comment': dis_name_descript['name_y'], 'classes': dis_name_descript['name_x']})
    comment_disease.to_csv('data/comment_disease.csv', index=False)

    pass

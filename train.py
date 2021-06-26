#  Copyright 2021 by Dmytro Vynnyk.
#  All rights reserved.
#  This file is part of the your-doc-bot Telegram bot,
#  and is released under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
#  To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/
#  or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#  Please see the LICENSE file that should have been included as part of this package.

# !pip install simpletransformers

# %load_ext tensorboard

# %tensorboard --logdir runs

# !find ./outputs -mindepth 1 ! -regex '^./outputs/best_model_bert\(/.*\)?' -delete

import math
from ast import literal_eval

import pandas as pd
import sklearn
from simpletransformers.classification import MultiLabelClassificationModel

if __name__ == '__main__':
    df = pd.read_csv('data/comment_specialty.csv')
    # df = pd.read_csv('/content/drive/MyDrive/DocHack/data/comment_specialty.csv')
    df['labels'] = df['labels'].apply(literal_eval)
    print(df.head())

    train_size = 0.8
    train_dataframe = df.sample(frac=train_size, random_state=200)
    test_dataframe = df.drop(train_dataframe.index).reset_index(drop=True)
    train_dataframe = train_dataframe.reset_index(drop=True)

    # GTX1080/32Gb (page file increased):
    #   bert-base-multilingual-uncased:24/182/3e-05  168M
    #   bert-base-multilingual-uncased:16/256/3e-05
    #   xlm-roberta-base:14/182/2e-05  270M, 2.5TB
    #   xlm-roberta-base:10/256/2e-05  550M, 2.5TB
    #   xlm-roberta-large:does not work

    batch_size = 14
    model = MultiLabelClassificationModel(
        'xlmroberta',  # bert, xlmroberta
        'xlm-roberta-base',  # bert-base-multilingual-uncased, xlm-roberta-base(large), outputs/best_model
        num_labels=31,
        args={
            'max_seq_length': 182,
            'overwrite_output_dir': True,
            'num_train_epochs': 5,

            'save_eval_checkpoints': False,
            'evaluate_during_training': True,
            'evaluate_during_training_verbose': True,
            'evaluate_during_training_steps': math.ceil(len(train_dataframe) / (batch_size * 4)),

            # 'early_stopping_patience': 3,
            # 'early_stopping_delta': 0,
            # 'early_stopping_metric': "eval_loss",
            # 'early_stopping_metric_minimize': True,
            # 'use_early_stopping': True,

            'save_steps': -1,
            'save_model_every_epoch': False,
            'evaluate_each_epoch': False,

            'train_batch_size': batch_size,
            'eval_batch_size': batch_size,

            'learning_rate': 2e-05,
        },
    )

    model.train_model(
        train_df=train_dataframe, eval_df=test_dataframe,
        f1=lambda a, b: sklearn.metrics.f1_score(a, b > model.args.threshold, average='micro')
    )

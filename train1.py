# !pip install simpletransformers

# %load_ext tensorboard

# %tensorboard --logdir runs

# !find ./outputs -mindepth 1 ! -regex '^./outputs/best_model\(/.*\)?' -delete

import math

import pandas as pd
from ast import literal_eval

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

    batch_size = 4  # xlm-roberta-base:48, xlm-roberta-large:10 -- p100, xlm-roberta-large:8 -- T4
    model = MultiLabelClassificationModel(
        "bert",  # xlmroberta
        "bert-base-multilingual-uncased",  # xlm-roberta-large, outputs/best_model
        num_labels=31,
        args={
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "num_train_epochs": 5,
            "max_seq_length": 256,  # 196 256

            "save_eval_checkpoints": False,
            "evaluate_during_training": True,
            "evaluate_during_training_verbose": True,
            "evaluate_during_training_steps": math.ceil(len(train_dataframe) / (batch_size * 4)),

            # "early_stopping_patience": 3,
            # "early_stopping_delta": 0,
            # "early_stopping_metric": "eval_loss",
            # "early_stopping_metric_minimize": True,
            # "use_early_stopping": True,

            "save_steps": -1,
            "save_model_every_epoch": False,
            "evaluate_each_epoch": False,

            "train_batch_size": batch_size,
            "eval_batch_size": batch_size,

            "learning_rate": 2e-05,  # bert: 4e-05, xlm-roberta-base: 2e-5, xlm-roberta-large: 1e-5
        },
    )

    model.train_model(train_df=train_dataframe, eval_df=test_dataframe,
                      f1=lambda a, b: sklearn.metrics.f1_score(a, b > 0.5, average='micro'))

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_dataframe)
    print(result)
    print(model_outputs)

    predictions, raw_outputs = model.predict(["This thing is entirely different from the other thing. "])
    print(predictions)
    print(raw_outputs)
# Copyright 2021 by Dmytro Vynnyk.
# All rights reserved.
# This file is part of the your-doc-bot Telegram bot,
# and is released under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/
# or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
# Please see the LICENSE file that should have been included as part of this package.

# !pip install torch_optimizer transformers SentencePiece

import datetime
import os
import time
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
import torch_optimizer as optim
from sklearn import metrics
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# noinspection PyUnresolvedReferences
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import BertPreTrainedModel
# noinspection PyUnresolvedReferences
from transformers import RobertaModel, XLMRobertaConfig, XLMRobertaTokenizerFast
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

writer = SummaryWriter()

DEVICE = 'cuda'
MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 10e-06

# MODEL = 'RobertaForMultiLabelSequenceClassification'
# MODEL_TYPE = 'xlm-roberta-base'
# CONFIG = XLMRobertaConfig
# TOKENIZER = XLMRobertaTokenizerFast

MODEL = 'BertForMultiLabelSequenceClassification'
MODEL_TYPE = 'bert-base-multilingual-uncased'
CONFIG = BertConfig
TOKENIZER = BertTokenizer

CLASSES_NUM = 31


class ClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        text_a, labels = (data.iloc[:, 0].astype(str).tolist(), data.iloc[:, 1].tolist())

        self.examples = tokenizer(
            text=text_a,
            text_pair=None,
            truncation=True,
            padding='max_length',
            max_length=max_seq_length,
            return_tensors='pt',
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples['input_ids'])

    def __getitem__(self, index):
        return {key: self.examples[key][index] for key in self.examples}, self.labels[index]


class RobertaForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, pos_weight=None):
        super(RobertaForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, pos_weight=None):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 0.1
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.pos_weight = pos_weight

        self.init_weights()

    def forward(
            self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def validation(log_steps=50):
    model.eval()
    targets = []
    preds = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            targets_i = batch[1]
            inputs = {key: value.squeeze(1).to(DEVICE) for key, value in batch[0].items()}
            inputs['labels'] = targets_i.to(DEVICE)

            outputs_i = model(**inputs)
            logits_i = outputs_i[1]

            targets.extend(targets_i.cpu().detach().numpy())
            preds.extend(torch.sigmoid(logits_i).cpu().detach().numpy())

            if i % log_steps == 0:
                if i > 0:
                    tac = time.time()
                    secs = (len(test_loader) - i) * (tac - tic) / log_steps
                    print(f'Validation: {i}/{len(test_loader)} '
                          f'Secs: {secs:.2f}, Mins: {secs / 60:.2f}, Hours: {secs / 60 / 60:.2f}')
                tic = time.time()

    pred = np.vstack(preds) > 0.5
    accuracy = metrics.accuracy_score(targets, pred)
    f1_score_micro = metrics.f1_score(targets, pred, average='micro')
    f1_score_macro = metrics.f1_score(targets, pred, average='macro')
    print(f'Accuracy Score = {accuracy}')  # % of full matches
    print(f'F1 Score (Micro) = {f1_score_micro}')
    print(f'F1 Score (Macro) = {f1_score_macro}')  # all classes are equal important
    return f1_score_micro


def train(epoch, log_steps=135):
    model.train()

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        targets = batch[1]
        inputs = {key: value.squeeze(1).to(DEVICE) for key, value in batch[0].items()}
        inputs['labels'] = targets.to(DEVICE)

        outputs = model(**inputs)
        loss = outputs[0]
        pred = torch.sigmoid(outputs[1]).cpu()

        acc = metrics.accuracy_score(targets, pred > 0.5)
        if i % log_steps == 0:
            if i > 0:
                tac = time.time()
                # noinspection PyUnboundLocalVariable
                secs = (len(train_loader) - i) * (tac - tic) / log_steps
                print(f'Secs: {secs:.2f}, Mins: {secs / 60:.2f}, Hours: {secs / 60 / 60:.2f}, '
                      f'Epoch: {epoch}: {i}/{len(train_loader)}, '
                      f'Loss: {loss}, '
                      f'Acc: {acc}')
                writer.add_scalar('Loss/train', loss, global_step=i)
            tic = time.time()

        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    df = pd.read_csv('data/comment_specialty.csv')
    # df = pd.read_csv('/content/drive/MyDrive/DocHack/data/comment_specialty.csv')
    df['labels'] = df['labels'].apply(literal_eval)
    print(df.head())

    train_size = 0.8
    train_dataframe = df.sample(frac=train_size, random_state=200)
    test_dataframe = df.drop(train_dataframe.index).reset_index(drop=True)
    train_dataframe = train_dataframe.reset_index(drop=True)

    print(f'Full  size: {df.shape}')
    print(f'Train size: {train_dataframe.shape}')
    print(f'Test  size: {test_dataframe.shape}')

    tokenizer_ = TOKENIZER.from_pretrained(MODEL_TYPE, cache_dir='cache')
    train_dataset = ClassificationDataset(train_dataframe, tokenizer_, MAX_LEN)
    test_dataset = ClassificationDataset(test_dataframe, tokenizer_, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True)

    config = CONFIG.from_pretrained(MODEL_TYPE, num_labels=CLASSES_NUM, hidden_dropout_prob=0.3)
    model = eval(MODEL)(config)
    model.to(DEVICE)

    # optimizer = optim.RAdam(params=model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.AdaBound(params=model.parameters(), lr=LEARNING_RATE)

    os.makedirs('checkpoints', exist_ok=True)

    # model = torch.load('checkpoints/model6_loss_0.0648547112941742')
    max_f1 = 0.
    model_ver = 1
    for e in range(1000):
        log_dir = f'logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{e:04d}'
        writer = SummaryWriter(log_dir)

        train(epoch=e)
        f1_micro = validation()

        writer.add_scalar('F1 Micro', f1_micro, global_step=e)

        if f1_micro > max_f1:
            max_f1 = f1_micro
            file_name = f'checkpoints/modelV{model_ver}_E{e:04d}_F{f1_micro:.4f}.pth'
            torch.save(model, file_name)
            print('Save', file_name)
        else:
            print(f'Skip {f1_micro} < {max_f1}')

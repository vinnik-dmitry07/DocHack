# Copyright 2021 by Dmytro Vynnyk.
# All rights reserved.
# This file is part of the your-doc-bot Telegram bot,
# and is released under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/
# or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.".
# Please see the LICENSE file that should have been included as part of this package.

import time
from ast import literal_eval
from typing import List

import numpy as np
import pandas as pd
import torch
import torch_optimizer as optim
import transformers
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

DEVICE = 'cuda'
MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 10e-06
MODEL = transformers.BertModel  # Multilabel classification
MODEL_TYPE = 'bert-base-multilingual-uncased'
MODEL_LAST_LAYER_WIDTH = 768
TOKENIZER = transformers.BertTokenizer
CLASSES_NUM = 31


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.comment = dataframe.comment
        self.targets = self.dataframe['classes']
        self.max_len = max_len

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, index):
        comment = self.comment[index]

        inputs = self.tokenizer.encode_plus(
            comment,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids: List[int] = inputs['input_ids']
        mask: List[int] = inputs['attention_mask']
        token_type_ids: List[int] = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = MODEL.from_pretrained(MODEL_TYPE, cache_dir='cache')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(MODEL_LAST_LAYER_WIDTH, CLASSES_NUM)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def validation(log_steps=50):
    model.eval()
    targets = []
    outputs = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            ids = data['ids'].to(DEVICE)
            mask = data['mask'].to(DEVICE)
            token_type_ids = data['token_type_ids'].to(DEVICE)
            targets_i = data['targets'].to(DEVICE)

            outputs_i = model(ids, mask, token_type_ids)

            targets.extend(targets_i.cpu().detach().numpy().tolist())
            outputs.extend(torch.sigmoid(outputs_i).cpu().detach().numpy().tolist())

            if i % log_steps == 0:
                if i > 0:
                    tac = time.time()
                    secs = (len(test_loader) - i) * (tac - tic) / log_steps
                    print(f'Validation: {i}/{len(test_loader)} '
                          f'Secs: {secs:.2f}, Mins: {secs / 60:.2f}, Hours: {secs / 60 / 60:.2f}')
                tic = time.time()

    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f'Accuracy Score = {accuracy}')  # % of full matches
    print(f'F1 Score (Micro) = {f1_score_micro}')
    print(f'F1 Score (Macro) = {f1_score_macro}')  # all classes are equal important
    return f1_score_micro


def train(epoch, log_steps=5, valid_num=10, model_ver=8):
    model.train()
    max_f1 = 0.

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        ids = data['ids'].to(DEVICE)
        mask = data['mask'].to(DEVICE)
        token_type_ids = data['token_type_ids'].to(DEVICE)
        targets = data['targets'].to(DEVICE)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)

        acc = metrics.accuracy_score(data['targets'], torch.sigmoid(outputs).cpu() > 0.5)
        if i % log_steps == 0:
            if i > 0:
                tac = time.time()
                # noinspection PyUnboundLocalVariable
                secs = (len(train_loader) - i) * (tac - tic) / log_steps
                print(f'Secs: {secs:.2f}, Mins: {secs / 60:.2f}, Hours: {secs / 60 / 60:.2f}')
                # noinspection PyUnboundLocalVariable
                print(f'Epoch: {epoch}: {i}/{len(train_loader)}, '
                      f'Loss: {loss}, '
                      f'Acc: {acc}')
                writer.add_scalar('Loss/train', loss, global_step=i)
            tic = time.time()

        if i > 0 and i % max(1, len(train_loader) // valid_num) == 0:
            f1 = validation()
            if f1 > max_f1:
                max_f1 = f1
                file_name = f'checkpoints/modelV{model_ver}_E{epoch:02d}_S{i:04d}_F{f1:.4f}.pth'
                torch.save(model, file_name)
                print('Save', file_name)
            else:
                print(f'Skip {f1} < {max_f1}')

        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    df = pd.read_csv('data/comment_specialty.csv')
    df['classes'] = df['classes'].apply(literal_eval)
    print(df.head())

    tokenizer_ = TOKENIZER.from_pretrained(MODEL_TYPE, cache_dir='cache')

    train_size = 0.8
    train_dataframe = df.sample(frac=train_size, random_state=200)
    test_dataframe = df.drop(train_dataframe.index).reset_index(drop=True)
    train_dataframe = train_dataframe.reset_index(drop=True)

    print(f'Full  size: {df.shape}')
    print(f'Train size: {train_dataframe.shape}')
    print(f'Test  size: {test_dataframe.shape}')

    train_dataset = CustomDataset(train_dataframe, tokenizer_, MAX_LEN)
    test_dataset = CustomDataset(test_dataframe, tokenizer_, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True)

    model = BERTClass()
    model.to(DEVICE)

    # optimizer = optim.RAdam(params=model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.AdaBound(params=model.parameters(), lr=LEARNING_RATE)

    # model = torch.load('checkpoints/model6_loss_0.0648547112941742')
    for e in range(EPOCHS):
        train(epoch=e)

# Copyright 2021 by Dmytro Vynnyk.
# All rights reserved.
# This file is part of the your-doc-bot Telegram bot,
# and is released under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/
# or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.".
# Please see the LICENSE file that should have been included as part of this package.

import time

import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from ast import literal_eval
import torch_optimizer as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

writer = SummaryWriter()

DEVICE = 'cuda'
MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 10e-06
MODEL = transformers.BertModel
MODEL_TYPE = 'bert-base-multilingual-uncased'
MODEL_LAST_LAYER_WIDTH = 768
TOKENIZER = transformers.BertTokenizer
CLASSES_NUM = 12


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment = dataframe.comment
        self.targets = self.data['classes']
        self.max_len = max_len

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, index):
        comment = self.comment[index]

        inputs = self.tokenizer.encode_plus(
            comment,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

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
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train(epoch, log_steps=5, saves_num=40, model_ver=5):
    model.train()
    min_loss = 0.3
    max_acc = 0.25

    for i, data in enumerate(training_loader):
        ids = data['ids'].to(DEVICE, dtype=torch.long)
        mask = data['mask'].to(DEVICE, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
        targets = data['targets'].to(DEVICE, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        acc = metrics.accuracy_score(data['targets'], outputs.cpu() > 0.5)

        if loss < min_loss:
            min_loss = loss
            torch.save(model, f'checkpoints/model{model_ver}_loss_{min_loss}')
            print(f'Save Loss: {loss}, Acc: {acc}')

        if acc > max_acc:
            max_acc = acc
            torch.save(model, f'checkpoints/model{model_ver}_acc_{max_acc}')
            print(f'Save Loss: {loss}, Acc: {acc}')

        if i % log_steps == 0:
            if i > 0:
                tac = time.time()
                # noinspection PyUnboundLocalVariable
                secs = (len(training_loader) - i) * (tac - tic) / log_steps
                print(f'Secs: {secs:.2f}, Mins: {secs / 60:.2f}, Hours: {secs / 60 / 60:.2f}')
                # noinspection PyUnboundLocalVariable
                print(f'Epoch: {epoch}: {i}/{len(training_loader)}, '
                      f'Loss: {loss}, '
                      f'Acc: {acc}')
                writer.add_scalar('Loss/train', loss, global_step=i)
            tic = time.time()

        if i % max(1, len(training_loader) // saves_num) == 0:
            torch.save(model, f'checkpoints/model{model_ver}_{epoch}')
            print('Save', i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    df = pd.read_csv('comment_classes.csv')
    df['classes'] = df['classes'].apply(literal_eval)
    new_df = df.copy()
    print(new_df.head())

    tokenizer_ = TOKENIZER.from_pretrained(MODEL_TYPE, cache_dir='cache')  # bert-base-multilingual-uncased

    train_size = 0.8
    train_dataset = new_df.sample(frac=train_size, random_state=200)
    test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print(f'FULL Dataset: {new_df.shape}')
    print(f'TRAIN Dataset: {train_dataset.shape}')
    print(f'TEST Dataset: {test_dataset.shape}')

    training_set = CustomDataset(train_dataset, tokenizer_, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer_, MAX_LEN)

    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    test_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = BERTClass()
    model.to(DEVICE)

    # optimizer = optim.RAdam(params=model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.AdaBound(params=model.parameters(), lr=LEARNING_RATE)

    # model = torch.load('checkpoints/model4_0')
    for epoch_ in range(EPOCHS):
        train(epoch_)

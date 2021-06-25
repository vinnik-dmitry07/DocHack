import math
import os

import pandas as pd
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, RandomSampler, DataLoader
from transformers import (
    BertModel, BertPreTrainedModel, BertTokenizer, BertConfig,
    get_linear_schedule_with_warmup,
)


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Bert model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
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


class ClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        text_a, labels = data

        self.examples = tokenizer(
            text=text_a,
            text_pair=None,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        return {key: self.examples[key][index] for key in self.examples}, self.labels[index]


train_df = pd.read_csv('data/comment_specialty.csv')
# XLMRobertaConfig, XLMRobertaForMultiLabelSequenceClassification, XLMRobertaTokenizer
model_name = 'bert-base-multilingual-uncased'
config = BertConfig.from_pretrained(model_name, num_labels=31)
model = BertForMultiLabelSequenceClassification.from_pretrained(model_name, config=config)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.to('cuda')

train_examples = (train_df.iloc[:, 0].astype(str).tolist(), train_df.iloc[:, 1].tolist())
train_dataset = ClassificationDataset(train_examples, tokenizer, max_seq_length=200)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=8,
    num_workers=0,
)

os.makedirs('checkpoints', exist_ok=True)

num_train_epochs = 5
warmup_ratio = 0.06
t_total = len(train_dataloader) // num_train_epochs
warmup_steps = math.ceil(t_total * warmup_ratio)

optimizer =
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

from torch.cuda import amp
scaler = amp.GradScaler()
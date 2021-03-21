import logging
from functools import lru_cache

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
from telegram.ext import Updater

from prepare_data import pre_process
from train import BERTClass, TOKENIZER, MODEL_TYPE, DEVICE, MAX_LEN

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

model = BERTClass()
model.to(DEVICE)
model = torch.load('checkpoints/model4_acc_0.75')
model.eval()

tokenizer_ = TOKENIZER.from_pretrained(MODEL_TYPE, cache_dir='cache')

_labels = [3, 13, 52, 15, 56, 61, 8, 23, 28, 40, 49, 53]
labels = ['Акушер-гинеколог', 'Гинеколог', 'УЗИ-специалист', 'Дерматолог', 'Хирург', 'Дерматовенеролог',
          'Гастроэнтеролог', 'Отоларинголог (ЛОР)', 'Невролог', 'Психотерапевт', 'Терапевт', 'Уролог']
mlb = MultiLabelBinarizer(sparse_output=False, classes=labels)


@lru_cache(maxsize=None)
def get_prediction(text_input):
    inputs = tokenizer_.encode_plus(
        text_input,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        return_token_type_ids=True
    )

    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    # We have to add another dimension to the tensors
    # hence the [ids], [mask], etc...
    ids = torch.tensor([ids], dtype=torch.long)
    mask = torch.tensor([mask], dtype=torch.long)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)

    ids = ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    outputs = model(ids, mask, token_type_ids)
    outputs = outputs > 0.5

    # fit possible labels to y
    mlb.fit(outputs)

    # Decode prediction array (returns tuple with categories)
    prediction = mlb.inverse_transform(outputs.cpu())[0]
    return prediction


updater = Updater(token='1676533113:AAFA67p-IV-ggKnqHDxJCv3UZuz7EXfRjJs', use_context=True)
dispatcher = updater.dispatcher


def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text='Привіт, що болить?')


start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)


def answer(update, context):
    an_answer = ' '.join(get_prediction(pre_process(update.message.text)))
    context.bot.send_message(chat_id=update.effective_chat.id, text=an_answer)


handler = MessageHandler(Filters.text & (~Filters.command), answer)
dispatcher.add_handler(handler)

updater.start_polling()

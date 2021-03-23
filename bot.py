# Copyright 2021 by Dmytro Vynnyk.
# All rights reserved.
# This file is part of the your-doc-bot Telegram bot,
# and is released under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/
# or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.".
# Please see the LICENSE file that should have been included as part of this package.

from typing import Tuple

import pandas as pd
import scipy.spatial
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, MessageHandler, Filters, Updater, CallbackContext, CallbackQueryHandler, \
    ConversationHandler

from prepare_data import pre_process
from secret import TOKEN
from train import BERTClass, TOKENIZER, MODEL_TYPE, DEVICE, MAX_LEN

# import logging
# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

DISEASE_OPTION, DOCTOR_OPTION, SELECTING_HANDLER = map(str, range(3))
OPTION_PATTERN_PREFIX = 'option_'


def get_prediction(text_input) -> Tuple[str]:
    inputs = tokenizer.encode_plus(
        text_input,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=True
    )

    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    ids = torch.tensor([ids], dtype=torch.long)
    mask = torch.tensor([mask], dtype=torch.long)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)

    ids = ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    outputs = model(ids, mask, token_type_ids)
    outputs = torch.sigmoid(outputs) > 0.5

    mlb.fit(outputs)

    prediction = mlb.inverse_transform(outputs.cpu())[0]
    return prediction


def answer_disease(update: Update, context: CallbackContext) -> str:
    if not update.message:
        return start(update, context)

    query = pre_process(update.message.text)
    query_embedding = embedder.encode([query])

    closest_n = 5
    distances = scipy.spatial.distance.cdist(query_embedding, corpus_embeddings, 'cosine')[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    res = 'Найбільш схожі хвороби: \n'
    for idx, distance in results[0:closest_n]:
        res += disease_symptoms['disease'][idx] + f' (Імовірність: {1 - distance:.4f})\n'

    update.message.reply_text(res)

    return start(update, context)


def answer_doctor(update: Update, context: CallbackContext) -> str:
    if not update.message:
        return start(update, context)

    answer = ' '.join(get_prediction(pre_process(update.message.text)))
    answer = answer if answer else 'Нажаль цієї інформації не достатньо :('
    update.message.reply_text(answer)

    return start(update, context)


def start(update: Update, _) -> str:
    keyboard = [[
        InlineKeyboardButton('Визначити хворобу', callback_data=OPTION_PATTERN_PREFIX + DISEASE_OPTION),
        InlineKeyboardButton('Визначити лікаря', callback_data=OPTION_PATTERN_PREFIX + DOCTOR_OPTION),
    ]]

    reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text('Виберіть опцію:', reply_markup=reply_markup)
    return SELECTING_HANDLER


def stop(update: Update, _) -> int:
    update.message.reply_text('Ок, пока.')
    return ConversationHandler.END


def button(update: Update, context: CallbackContext) -> str:
    query = update.callback_query
    query.answer()
    context.bot.send_message(chat_id=update.effective_chat.id, text='Опишіть симптоми:')
    return query.data.strip(OPTION_PATTERN_PREFIX)


def main() -> None:
    updater = Updater(token=TOKEN, use_context=True)

    default_handlers = [CommandHandler('start', start),
                        CallbackQueryHandler(button, pattern='^' + OPTION_PATTERN_PREFIX)]
    # noinspection PyTypeChecker
    conv_handler = ConversationHandler(
        entry_points=default_handlers,
        states={
            SELECTING_HANDLER: default_handlers,
            DISEASE_OPTION: [MessageHandler(Filters.text & (~Filters.command), answer_disease)] + default_handlers,
            DOCTOR_OPTION: [MessageHandler(Filters.text & (~Filters.command), answer_doctor)] + default_handlers,
        },
        fallbacks=[CommandHandler('stop', stop)],
    )

    updater.dispatcher.add_handler(conv_handler)
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    model = BERTClass()
    model.to(DEVICE)
    model = torch.load('checkpoints/modelV7_E4_S2682_F0.4749.pth')
    model.eval()

    tokenizer = TOKENIZER.from_pretrained(MODEL_TYPE, cache_dir='cache')

    speciality_names = pd.read_csv('data/speciality_id_name.csv')['name'].tolist()
    mlb = MultiLabelBinarizer(classes=speciality_names)

    disease_symptoms = pd.read_csv('data/disease_symptoms.csv')
    corpus = disease_symptoms['symptoms'].tolist()
    embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')  # semantic search
    corpus_embeddings = embedder.encode(corpus)

    main()

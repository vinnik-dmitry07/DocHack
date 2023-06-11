from typing import Tuple

import numpy as np
import pandas as pd
import scipy.spatial
from sentence_transformers import SentenceTransformer
from simpletransformers.classification import MultiLabelClassificationModel
from sklearn.preprocessing import MultiLabelBinarizer
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, MessageHandler, Filters, Updater, CallbackContext, CallbackQueryHandler, \
    ConversationHandler

from prepare_data import pre_process
from secret import TOKEN

# import logging
# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

DISEASE_OPTION, DOCTOR_OPTION, SELECTING_HANDLER = map(str, range(3))
OPTION_PATTERN_PREFIX = 'option_'


def get_prediction(text_input) -> Tuple[str]:
    output = np.array(model.predict([text_input])[0])
    mlb.fit(output)
    prediction = mlb.inverse_transform(output)[0]
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

    res = 'The most similar diseases: \n'  # Найбільш схожі хвороби:
    for idx, distance in results[0:closest_n]:
        res += disease_symptoms['disease'][idx] + f' (Probability: {1 - distance:.4f})\n'  # Імовірність

    update.message.reply_text(res)

    return start(update, context)


def answer_doctor(update: Update, context: CallbackContext) -> str:
    if not update.message:
        return start(update, context)

    answer = ' '.join(get_prediction(pre_process(update.message.text)))
    answer = answer if answer else 'Sorry, this information is not enough :('  # На жаль, цієї інформації недостатньо :(
    update.message.reply_text(answer)

    return start(update, context)


def start(update: Update, _) -> str:
    keyboard = [[
        # Визначити хворобу
        InlineKeyboardButton('Identify the disease', callback_data=OPTION_PATTERN_PREFIX + DISEASE_OPTION),
        # Визначити лікаря
        InlineKeyboardButton('Identify the doctor', callback_data=OPTION_PATTERN_PREFIX + DOCTOR_OPTION),
    ]]

    reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text('Select an option:', reply_markup=reply_markup)  # Виберіть опцію:
    return SELECTING_HANDLER


def stop(update: Update, _) -> int:
    update.message.reply_text('Ok, bye.')  # Ок, пока.
    return ConversationHandler.END


def button(update: Update, context: CallbackContext) -> str:
    query = update.callback_query
    query.answer()
    context.bot.send_message(chat_id=update.effective_chat.id, text='Describe the symptoms:')  # Опишіть симптоми^
    return query.data.strip(OPTION_PATTERN_PREFIX)


def main() -> None:
    updater = Updater(token=TOKEN, use_context=True)

    default_handlers = [
        CommandHandler('start', start),
        CallbackQueryHandler(button, pattern='^' + OPTION_PATTERN_PREFIX)
    ]
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
    model = MultiLabelClassificationModel(
        'xlmroberta',  # bert, xlmroberta
        'outputs/best_model_roberta_punct',
        num_labels=31,
        args={
            'max_seq_length': 182,
            'use_multiprocessing_for_evaluation': False,
        }
    )

    speciality_names = pd.read_csv('data/speciality_id_name.csv')['name'].tolist()
    mlb = MultiLabelBinarizer(classes=speciality_names)

    disease_symptoms = pd.read_csv('data/disease_symptoms.csv')
    corpus = disease_symptoms['symptoms'].tolist()
    # paraphrase-xlm-r-multilingual-v1 -- better
    # distiluse-base-multilingual-cased-v2
    embedder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')  # semantic search
    corpus_embeddings = embedder.encode(corpus)

    main()

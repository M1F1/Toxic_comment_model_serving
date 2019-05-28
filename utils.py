import os
import tensorflow as tf
import json
import pandas as pd
import numpy as np
import pickle
from data_preparation import clean_string
import requests


def query_for_answers(X_test, y_test, SERVER_URL, batch_size=16):
    good_answers = 0

    for i in list(range(0, X_test.shape[0], batch_size)):
        X_query = X_test[i:(i + batch_size)]
        y_query = y_test[i:(i + batch_size)]

        input_data_json = json.dumps({
            "signature_name": "serving_default",
            "instances": X_query.tolist(),
        })

        response = requests.post(SERVER_URL, data=input_data_json)
        response.raise_for_status()
        response = response.json()

        y_proba = np.array(response["predictions"])
        good_answers += np.sum(np.argmax(y_proba, axis=-1) == y_query)

    return good_answers / X_test.shape[0]


def load_data(filename='data/test.txt', separator=';'):
    texts = []
    labels = []
    with open(filename, mode='r') as handle:
        for line in handle:
            values = line.split(separator)
            texts.append(values[0])
            labels.append(int(values[1].rstrip()))

    return texts, labels


def prepare_texts(texts):

    with open('config.json', 'r') as f:
        config = json.load(f)
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print('tokenizer was loaded')
    stopwords_path = os.path.join(os.getcwd(), 'data', 'stopwords.txt')
    texts = [clean_string(text, stopwords_path) for text in texts]
    texts = tokenizer.texts_to_sequences(texts)
    texts = tf.keras.preprocessing.sequence.pad_sequences(texts, maxlen=config['MAX_SEQUENCE_LENGTH'])
    return texts


if __name__ == '__main__':

    all_models_path = 'models'
    model_name = "cnn"
    model_path = os.path.join(all_models_path, model_name, '1558622633')
    model_saved_path = os.path.join(model_path, 'model.h5')
    model = tf.keras.models.load_model(model_saved_path)
    input_name = model.input_names[0]
    model_tf_serving_path = os.path.join(model_path, 'tf_serving_model')
    print('input_name:', input_name)
    print(model_tf_serving_path)

    with open('config.json', 'r') as f:
        config = json.load(f)

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print('tokenizer was loaded')
    stopwords_path = os.path.join(os.getcwd(), 'data', 'stopwords.txt')


    texts, labels = load_data()
    print(texts)
    texts = prepare_texts(texts)


    SERVER_URL = 'http://localhost:8501/v1/models/cnn:predict'
    acc = query_for_answers(texts, labels, SERVER_URL)
    print(acc)

    # p = model.predict(texts)
    # print(p)
    # run docker sudo docker run -it --rm -p 8501:8501 -v "`pwd`/models/cnn:/models/cnn" -e MODEL_NAME=cnn tensorflow/serving



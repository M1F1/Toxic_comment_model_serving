import os
import tensorflow as tf
import json
import pandas as pd
import numpy as np
import pickle
from data_preparation import clean_string


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


def load_data(filename='data/test.txt', separator=';'):
    texts = []
    labels = []
    with open(filename, mode='r') as handle:
        for line in handle:
            values = line.split(separator)
            texts.append(values[0])
            labels.append(values[1])
    return texts, labels


def prepare_texts(texts):
    stopwords_path = os.path.join(os.getcwd(), 'data', 'stopwords.txt')
    texts = [clean_string(text, stopwords_path) for text in texts]
    texts = tokenizer.texts_to_sequences(texts)
    texts = tf.keras.preprocessing.sequence.pad_sequences(texts, maxlen=config['MAX_SEQUENCE_LENGTH'])
    return texts


texts, labels = load_data()
print(texts)
texts = prepare_texts(texts)
p = model.predict(texts)
print(p)
# import subprocess
# subprocess.call(["ls", "-l"])
#
# subprocess.call(["saved_model_cli", "run",
#                  "--dir",
#                  "{}".format(model_tf_serving_path),
#                  "--tag_set serve",
#                  "--signature_def serving_default",
#                  "--inputs",
#                  "{}=exemplary_tests.npy".format(input_name)])



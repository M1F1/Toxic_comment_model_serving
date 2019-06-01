import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time
import json
import sys
import pickle
from models_architectures import CNN
# Python ≥3.5 required
assert sys.version_info >= (3, 5)
# TensorFlow ≥2.0 required
assert tf.__version__ >= "2.0"

with open('config.json', 'r') as f:
    config = json.load(f)

df = pd.read_csv(config['TEXT_DATA_PATH'])
texts = (np.array(df['comment_text'].astype(str)))
labels = list(df['toxic'])

tokenizer = keras.preprocessing.text.Tokenizer(num_words=config['MAX_WORDS'])
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
word_index = tokenizer.word_index
print("unique words : {}".format(len(word_index)))

data = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=config['MAX_SEQUENCE_LENGTH'])
labels = keras.utils.to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# data = data[:1000]
# labels = labels[:1000]
nb_validation_samples = int(config['VALIDATION_SPLIT'] * data.shape[0])
x_train = data[:nb_validation_samples]
y_train = labels[:nb_validation_samples]
x_val = data[nb_validation_samples:-config['TEST_SAMPLES_NUM']]
y_val = labels[nb_validation_samples:-config['TEST_SAMPLES_NUM']]
x_test = data[-config['TEST_SAMPLES_NUM']:]
y_test = labels[-config['TEST_SAMPLES_NUM']:]

print('Shape of data train tensor:', x_train.shape)
print('Shape of label train tensor:', y_train.shape)
print('Shape of data val tensor:', x_val.shape)
print('Shape of label val tensor:', y_val.shape)
print('Shape of data test tensor:', x_test.shape)
print('Shape of label test tensor:', y_test.shape)

# create embeddings
# create embeddings, UNK is the mean of all known vectors
EMBEDDING_DIM = 100

embeddings_index = {}
UNK = np.zeros(EMBEDDING_DIM,)
with open(os.path.join(config['GLOVE_DIR'])) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        UNK = UNK + coefs
        embeddings_index[word] = coefs

UNK = UNK / len(embeddings_index)
print('Found %s word vectors.' % len(embeddings_index))
in_glove = 0
embedding_matrix = np.zeros((len(word_index) + 1, config['EMBEDDING_DIM']))
not_in_glove = []
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        in_glove += 1
        embedding_matrix[i] = embedding_vector
    else:
        not_in_glove.append(word)
        embedding_matrix[i] = UNK

print('Number of words in dictionary that have embeddings in glove:', in_glove)

def lr_schedule(epoch):
    lr = 1e-3
    if epoch >= 6:
        lr = 1e-5
    elif epoch >= 3:
        lr = 1e-4
    print('Learning rate: ', lr)
    return lr


cnn = CNN(config=config, embedding_matrix=embedding_matrix, word_index=word_index)
model = cnn.create_model()
adam = keras.optimizers.Adam(lr=lr_schedule(0),
                             beta_1=0.9,
                             beta_2=0.999,
                             epsilon=1e-08,
                             decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

print("Traning Model...")
model.fit(x_train,
          y_train,
          batch_size=config['BATCH_SIZE'],
          epochs=config['EPOCHS'],
          verbose=1,
          validation_data=(x_val, y_val),
          )
print("Evaluating Model...")
model.evaluate(x_test, y_test)

# Saving model to .h5 and converting to .pb for tf serving
print("Serializing Model...")
all_models_path = 'models'
model_name = "cnn"
model_version = int(time.time())
model_path = os.path.join(all_models_path, model_name, str(model_version))
os.makedirs(model_path)
model_saved_path = os.path.join(model_path, 'model.h5')
model.save(model_saved_path)
tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference
model = tf.keras.models.load_model(model_saved_path)
tf.keras.experimental.export_saved_model(model, os.path.join(model_path))


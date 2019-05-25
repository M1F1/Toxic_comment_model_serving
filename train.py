import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time
import json
import sys
import pickle

assert sys.version_info >= (3, 5) # Python ≥3.5 required
assert tf.__version__ >= "2.0"    # TensorFlow ≥2.0 required

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
embeddings_index = {}
with open(os.path.join(config['GLOVE_DIR'])) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))
in_glove = 0
embedding_matrix = np.zeros((len(word_index) + 1, config['EMBEDDING_DIM']))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        in_glove += 1
        embedding_matrix[i] = embedding_vector

print('Number of words in dictionary that have embeddings in glove:', in_glove)


# setting up model architecture
embedding_layer = keras.layers.Embedding(len(word_index) + 1,
                                         config['EMBEDDING_DIM'],
                                         weights=[embedding_matrix],
                                         input_length=config['MAX_SEQUENCE_LENGTH'],
                                         trainable=False)

inputs = keras.layers.Input(shape=(config['MAX_SEQUENCE_LENGTH'],), dtype='int32')
embedding = embedding_layer(inputs)
print(embedding.shape)

reshape = keras.layers.Reshape((config['MAX_SEQUENCE_LENGTH'], config['EMBEDDING_DIM'], 1))(embedding)
print(reshape.shape)

num_filters = config['NUM_FILTERS']
embedding_dim = config['EMBEDDING_DIM']
max_sequence_length = config['MAX_SEQUENCE_LENGTH']
filter_size_1 = config['FILTER_SIZE_1']
filter_size_2 = config['FILTER_SIZE_2']
filter_size_3 = config['FILTER_SIZE_3']

conv_0 = keras.layers.Conv2D(num_filters,
                             kernel_size=(filter_size_1, embedding_dim),
                             padding='valid',
                             kernel_initializer='normal',
                             activation='relu')(reshape)
conv_1 = keras.layers.Conv2D(num_filters,
                             kernel_size=(filter_size_2, embedding_dim),
                             padding='valid',
                             kernel_initializer='normal',
                             activation='relu')(reshape)
conv_2 = keras.layers.Conv2D(num_filters,
                             kernel_size=(filter_size_3, embedding_dim),
                             padding='valid',
                             kernel_initializer='normal',
                             activation='relu')(reshape)

maxpool_0 = keras.layers.MaxPool2D(pool_size=(max_sequence_length - filter_size_1 + 1, 1),
                                   strides=(1, 1),
                                   padding='valid')(conv_0)
maxpool_1 = keras.layers.MaxPool2D(pool_size=(max_sequence_length - filter_size_2 + 1, 1),
                                   strides=(1, 1),
                                   padding='valid')(conv_1)
maxpool_2 = keras.layers.MaxPool2D(pool_size=(max_sequence_length - filter_size_3 + 1, 1),
                                   strides=(1, 1),
                                   padding='valid')(conv_2)

concatenated_tensor = keras.layers.Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = keras.layers.Flatten()(concatenated_tensor)
dropout = keras.layers.Dropout(config['DROP'])(flatten)
output = keras.layers.Dense(units=2,
                            activation='softmax')(dropout)

# this creates a model that includes
model = keras.models.Model(inputs=inputs,
                           outputs=output)

adam = keras.optimizers.Adam(lr=1e-4,
                             beta_1=0.9,
                             beta_2=0.999,
                             epsilon=1e-08,
                             decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

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
all_models_path = 'models'
model_name = "cnn"
model_version = int(time.time())
model_path = os.path.join(all_models_path, model_name, str(model_version))
os.makedirs(model_path)
model_saved_path = os.path.join(model_path, 'model.h5')
model.save(model_saved_path)
tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference
model = tf.keras.models.load_model(model_saved_path)
tf.keras.experimental.export_saved_model(model, os.path.join(model_path, 'tf_serving_model'))


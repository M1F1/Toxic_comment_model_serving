import os
import sys
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D
from keras.models import Model
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import json

with open('config.json', 'r') as f:
    config = json.load(f)

df = pd.read_csv('data/new_labels_train.csv')
texts = (np.array(df['comment_text'].astype(str)))
labels = list(df['toxic'])
print(type(texts[0]))
tokenizer = Tokenizer(num_words=config['MAX_WORDS'])
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print("unique words : {}".format(len(word_index)))

data = pad_sequences(sequences, maxlen=config['MAX_SEQUENCE_LENGTH'])

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
print(labels)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(config['VALIDATION_SPLIT'] * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Shape of data train tensor:', x_train.shape)
print('Shape of label train tensor:', y_train.shape)
print('Shape of data train tensor:', x_val.shape)
print('Shape of label train tensor:', y_val.shape)

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
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print(in_glove)


embedding_layer = Embedding(len(word_index) + 1,
                            config['EMBEDDING_DIM'],
                            weights=[embedding_matrix],
                            input_length=config['MAX_SEQUENCE_LENGTH'],
                            trainable=False)

inputs = Input(shape=(config['MAX_SEQUENCE_LENGTH'],), dtype='int32')
embedding = embedding_layer(inputs)
print(embedding.shape)

reshape = Reshape((config['MAX_SEQUENCE_LENGTH'], config['EMBEDDING_DIM'], 1))(embedding)
print(reshape.shape)

num_filters = config['NUM_FILTERS']
embedding_dim = config['EMBEDDING_DIM']
max_sequence_length = config['MAX_SEQUENCE_LENGTH']
filter_size_1 = config['FILTER_SIZE_1']
filter_size_2 = config['FILTER_SIZE_2']
filter_size_3 = config['FILTER_SIZE_3']

conv_0 = Conv2D(num_filters,
                kernel_size=(filter_size_1, embedding_dim),
                padding='valid',
                kernel_initializer='normal',
                activation='relu')(reshape)
conv_1 = Conv2D(num_filters,
                kernel_size=(filter_size_2, embedding_dim),
                padding='valid',
                kernel_initializer='normal',
                activation='relu')(reshape)
conv_2 = Conv2D(num_filters,
                kernel_size=(filter_size_3, embedding_dim),
                padding='valid',
                kernel_initializer='normal',
                activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(max_sequence_length - filter_size_1 + 1, 1),
                      strides=(1,1),
                      padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(max_sequence_length - filter_size_2 + 1, 1),
                      strides=(1,1),
                      padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(max_sequence_length - filter_size_3 + 1, 1),
                      strides=(1,1),
                      padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(config['DROP'])(flatten)
output = Dense(units=2,
               activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs,
              outputs=output)

checkpoint = ModelCheckpoint('weights_cnn_sentece.hdf5',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')
adam = Adam(lr=1e-4,
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
          callbacks=[checkpoint],
          validation_data=(x_val, y_val))




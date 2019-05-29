from tensorflow import keras


class CNN:
    def __init__(self, config, embedding_matrix, word_index):
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
        self.inputs = inputs
        self.output = output

    def create_model(self):
        model = keras.models.Model(inputs=self.inputs,
                                   outputs=self.output)
        return model
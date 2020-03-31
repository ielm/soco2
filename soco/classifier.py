from __future__ import print_function

import os
import sys
import click
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import model_from_json
from keras.initializers import Constant

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, '/Users/ivan/Dropbox/Code/soco2/data/glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '/Users/ivan/Dropbox/Code/soco2/data/gender/simple')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2


def index_word_vectors():
    print('\nIndexing word vectors.')
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, "glove.6B.200d.txt")) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    print(f"Found {len(embeddings_index)} word vectors.")
    return embeddings_index


def process_text(data_dir=TEXT_DATA_DIR):
    # prepare text samples and their labels
    print('\nProcessing gender dataset.')
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                    with open(fpath, **args) as f:
                        t = f.read()
                        i = t.find('\n\n')  # skip header
                        if 0 < i:
                            t = t[i:]
                        texts.append(t)
                    labels.append(label_id)
    print(f"Found {len(texts)} texts.")
    return texts, labels_index, labels


def vectorize(texts):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print(f"Found {len(word_index)} unique tokens.")
    return sequences, word_index


def label(sequences, labels):
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    # print('Shape of data tensor:', data.shape)
    # print('Shape of label tensor:', labels.shape)
    return data, labels


def split_train_val(data, labels):
    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    print('\nPreparing embedding matrix.')
    return x_train, y_train, x_val, y_val


def shuffle_master_set(data, labels):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    return data, labels


def load_embedding_layer(num_words, word_index, embeddings_index):
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return embedding_layer


def fit_model(save: bool = True):
    embeddings_index = index_word_vectors()
    # list of text samples | dictionary mapping label name to numeric id | list of label ids
    texts, labels_index, labels = process_text(data_dir=TEXT_DATA_DIR)
    # vectorize the text samples into a 2D integer tensor
    sequences, word_index = vectorize(texts)
    data, labels = label(sequences, labels)
    x_train, y_train, x_val, y_val = split_train_val(data, labels)
    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_layer = load_embedding_layer(num_words, word_index, embeddings_index)
    print('\nTraining model.')
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=20,
              validation_data=(x_val, y_val))
    loss, accuracy = model.evaluate(x=x_val,
                                    y=y_val,
                                    batch_size=128)
    print(f"Loss: {int(loss)}")
    print(f"Accuracy: {int(accuracy * 100)}")
    if save:
        serialize_to_json(model)
    return model


def serialize_to_json(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("../model/gender/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../model/gender/model.h5")
    print("Saved model to disk")


def load_from_json(path: str = "/Users/ivan/Dropbox/Code/soco2/model/gender"):
    # load json and create model
    if os.path.exists(f"{path}/model.json"):
        json_file = open(f"{path}/model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(f"{path}/model.h5")
        print("Loaded model from disk")
        return loaded_model
    else:
        print("\nCannot load model.\n"
              "Try running:  'python soco.py train' first.")
        exit()


if __name__ == '__main__':
    classifier = fit_conv1d_model()

# Reference: https://github.com/xitizzz/Text-CNN-Toxic-Comment-Classification/blob/master/Bi-LSTM.ipynb
import numpy as np
import pandas as pd

import gzip
import os
import gc
import gensim

from keras.models import Sequential, load_model
from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Dropout, Activation, Embedding
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.layers import Bidirectional, LSTM

hyperparam = {'sequence_len': 100,
              'embedding_dim': 300,
              'lstm_units': 100,
              'dropout' : 0.5,
              'batch_size': 256,
              'epochs': 1000,
              'steps_per_epochs': None,
              'early_stopping': True,
              'vocab_size': 193265,
              'learning_rate' : 0.001,
              'gradient_clip_value' : None,
              'gradient_clip_norm' : None,
              'validation_split': 0.2,
              'missing_word_vectors': 'normal',
              'dense_activation':'relu',
              'n_class': 6}

name = '_'.join(['Bi-LSTM',
                 str(hyperparam['sequence_len']),
                 str(hyperparam['lstm_units']),
                 str(hyperparam['batch_size']),
                 str(int(hyperparam['dropout']*100))])

save_predictions = True
save_model = False
use_best_checkpoint = True

filename = 'GoogleNews-vectors-negative300.bin'
word_vec = KeyedVectors.load_word2vec_format(filename, binary=True)

tokenizer = Tokenizer(num_words=hyperparam['vocab_size'], filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'')
train = pd.read_csv('train.csv')
train_text = train['comment_text'].astype('str').values
tokenizer.fit_on_texts(train_text)
train_seq = tokenizer.texts_to_sequences(train_text)
X_train = pad_sequences(train_seq, maxlen=hyperparam['sequence_len'], truncating='post', padding='post')
y_train = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=hyperparam['validation_split'], random_state=22)

if hyperparam['missing_word_vectors']=='normal':
    embed_list = []
    for word, index in tokenizer.word_index.items():
        if index >= hyperparam['vocab_size']:
            continue
        try:
            embed_list.append(word_vec.wv[word])
        except KeyError:
            pass
    a = np.array(embed_list)
    embedding_matrix = np.array(np.random.normal(a.mean(), a.std(), (hyperparam['vocab_size'], hyperparam['embedding_dim'])), dtype=np.float32)
    del embed_list
    del a
else:
    embedding_matrix = np.zeros((hyperparam['vocab_size'], hyperparam['embedding_dim']), dtype=np.float32)

if not os.path.exists(f'./{name}'):
    os.mkdir(f'./{name}')

with open(f'./{name}/Hyperparameters.csv', 'w') as file:
    file.write('{}, {}'.format('Hyperparameter', 'Value'))
    for key in hyperparam.keys():
        file.write('\n {}, {}'.format(str(key), str(hyperparam[key])))

def mini_batch_generator():
    global X_train, y_train
    while True:
        yield (X_train[:hyperparam['batch_size'], :], y_train[:hyperparam['batch_size'], :])
        X_train = np.roll(X_train, axis=0, shift=-hyperparam['batch_size'])
        y_train = np.roll(y_train, axis=0, shift=-hyperparam['batch_size'])

def computation_graph():
    model  = Sequential()
    model.add(Embedding(hyperparam['vocab_size'], hyperparam['embedding_dim'], weights=[embedding_matrix], name='Embedding_Layer'))
    model.add(Bidirectional(LSTM(units=100)))
    model.add(Dropout(rate=hyperparam['dropout'], name = 'Dropout_' + str(hyperparam['dropout'])))
    model.add(Dense(units=hyperparam['n_class'], activation='sigmoid', name='Dense_'+str(hyperparam['n_class'])+'_Sigmoid'))
    return model

model = computation_graph()

if hyperparam['gradient_clip_norm'] is None and hyperparam['gradient_clip_value'] is None:
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(hyperparam['learning_rate']),
                  metrics=['accuracy'])
elif hyperparam['gradient_clip_norm'] is None:
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(hyperparam['learning_rate'],
                  clipvalue=hyperparam['gradient_clip_value']),
                  metrics=['accuracy'])
elif hyperparam['gradient_clip_value'] is None:
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(hyperparam['learning_rate'],
                  clipnorm = hyperparam['gradient_clip_norm']),
                  metrics=['accuracy'])
else:
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(hyperparam['learning_rate'],
                  clipvalue=hyperparam['gradient_clip_value'],
                  clipnorm = hyperparam['gradient_clip_norm']),
                  metrics=['accuracy'])

if hyperparam['early_stopping']:
    callback = [EarlyStopping(verbose=1), ModelCheckpoint(f'./{name}/model_best.h5', save_best_only=True)]
    if hyperparam['steps_per_epochs']:
        callback = [EarlyStopping(verbose=1, patience=5), ModelCheckpoint(f'./{name}/model_best.h5', save_best_only=True)]
    validation_data = (X_val, y_val)
elif hyperparam['validation_split']:
    callback = [ModelCheckpoint(f'./{name}/model_best.h5', save_best_only=True)]
    validation_data = (X_val, y_val)
else:
    callback = None
    validation_data = None

if hyperparam['steps_per_epochs']:
        history = model.fit_generator(generator=mini_batch_generator(),
                              epochs=hyperparam['epochs'],
                              callbacks=callback,
                              validation_data = validation_data,
                              steps_per_epoch=hyperparam['steps_per_epochs'])
else:
        history = model.fit(x=X_train, y=y_train,
                          validation_data = validation_data,
                          epochs=hyperparam['epochs'],
                          batch_size=hyperparam['batch_size'],
                          shuffle=True,
                          callbacks=callback)

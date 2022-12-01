# Reference: https://github.com/xitizzz/Text-CNN-Toxic-Comment-Classification/blob/master/CNN_Advance_2_FC.ipynb
import numpy as np
import pandas as pd

import gzip
import os
import gc

from keras.models import Sequential, load_model
from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Dropout, Activation, Embedding
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec, KeyedVectors

hyperparam = {'sequence_len': 100,
              'embedding_dim': 300,
              'filters': 200,
              'kernel_size': 3,
              'dropout' : 0.5,
              'dense_units': 100,
              'batch_size': 512,
              'epochs': 1000,
              'steps_per_epochs': None,
              'early_stopping': True,
              'vocab_size': 193264,
              'learning_rate' : 0.0005,
              'gradient_clip_value' : None,
              'gradient_clip_norm' : None,
              'validation_split': 0.2,
              'missing_word_vectors': 'normal',
              'conv_activation': 'relu',
              'dense_activation':'relu',
              'n_class': 6}

name = '_'.join(['CNN_2_FC_Missing_relu',
                 str(hyperparam['sequence_len']),
                 str(hyperparam['filters']),
                 str(hyperparam['kernel_size']),
                 str(int(hyperparam['dropout']*100))])

save_predictions = True
save_model = False
use_best_checkpoint = True

filename = 'GoogleNews-vectors-negative300.bin'
word_vec = KeyedVectors.load_word2vec_format(filename, binary=True)

train = pd.read_csv('train.csv')
train_text = train['comment_text'].astype('str').values
tokenizer = Tokenizer(num_words=hyperparam['vocab_size'], filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'')
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
            embed_list.append(word_vec.key_to_index[word])
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
    model.add(Conv1D(filters=hyperparam['filters'],
                     kernel_size=hyperparam['kernel_size'],
                     activation = hyperparam['conv_activation'],
                     name= '_'.join(['Convolution_1D', str(hyperparam['filters']), str(hyperparam['kernel_size']), str(hyperparam['conv_activation'])])
                    ))
    model.add(GlobalMaxPool1D(name='Global_Max_Pooling'))
    model.add(Dense(units=hyperparam['filters'], name='Dense_'+str(hyperparam['filters'])))
    model.add(Dropout(rate=hyperparam['dropout'], name = 'Dropout_' + str(hyperparam['dropout'])))
    model.add(Dense(units=hyperparam['dense_units'], name='Dense_'+str(hyperparam['dense_units'])))
    model.add(Dropout(rate=hyperparam['dropout'], name = 'Dropout2_' + str(hyperparam['dropout'])))
    model.add(Activation(hyperparam['dense_activation'], name='Activation2_'+str(hyperparam['dense_activation'])))
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
    callback = [EarlyStopping(verbose=1, patience=3), ModelCheckpoint(f'./{name}/model_best.h5', save_best_only=True)]
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

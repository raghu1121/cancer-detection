import numpy as np

import pandas as pd
import tensorflow as tf
from keras.layers import Conv1D, Dropout, Flatten, Dense, AveragePooling1D
from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint,EarlyStopping

seed = 7
np.random.seed(seed)
df=pd.read_csv('dataset.csv')
X = df.loc[:,:'Ancestry_WHITE']
Y = df.loc[:,'Cancer type_Daisy':'Cancer type_control']

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
X_train, X_test,Y_train, Y_test = train_test_split(X,Y, stratify=Y, test_size=0.1, shuffle=True)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=False)
X_train.loc[:,'Age':'Max'] = scaler.fit_transform(X_train.loc[:,'Age':'Max'])
X_test.loc[:,'Age':'Max'] = scaler.transform(X_test.loc[:,'Age':'Max'])
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
verbose, epochs, batch_size = 0, 12, 30


def cnn1d_model(X_train, Y_train):
    n_timesteps, n_features, n_outputs = X_train.shape[0], X_train.shape[1], Y_train.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='selu', input_shape=(n_features,1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='selu'))
    model.add(Dropout(0.5))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation = (tf.nn.softmax)))
    return model

def evaluate_model(model, X_test, Y_test):
    _, accuracy,top2 = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=2)
    return [accuracy,top2]


def summarize_results(scores):
    print(scores)

import functools
top2_acc = functools.partial(top_k_categorical_accuracy, k=2)
top2_acc.__name__ = 'top2_acc'

earlystop = EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='auto')
filepath = "./models-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, \
                             save_best_only=True, save_weights_only=False, \
                             mode='auto', period=1)
callbacks=[checkpoint,earlystop]

# run an experiment
def run_experiment(repeats=10):
    # repeat experiment
    scores = list()
    for r in range(repeats):
        model = cnn1d_model(X_train, Y_train)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_test,Y_test), verbose=verbose,callbacks=callbacks)
        # [acc,top2] = evaluate_model(model, X_test, Y_test)
        # print(acc,top2)
        # evaluate model
        Y_test_pred = model.predict(X_test)
        predicted = np.argmax(Y_test_pred, axis=1)
        true = np.argmax(Y_test.to_numpy(), axis=1)

        report = classification_report(true, predicted)
        print(report)

run_experiment()
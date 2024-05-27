import pickle
import numpy as np
import os

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
actions = ['nothing', 'click', 'doubleclick', 'cap', 'alt', 'altf']


def train_video():
    data = np.zeros((1, 50, 100))
    for action in actions:
        f_r = open(f'dataset/{action}/last_idx.txt', 'r')
        action_count = int(f_r.readline())
        f_r.close()

        for i in range(action_count):
            seq_data = np.load(f'dataset/{action}/seq_{i + 1}.npy')
            if data.any() == 0:
                data = seq_data.reshape((1, 50, 100))
            else:
                print(action, i+1)
                print(seq_data.reshape((1, 50, 100))[:, 0, -1])
                data = np.append(data, seq_data.reshape((1, 50, 100)), axis=0)

    x_data = data[:, :, :-1]
    labels = data[:, 0, -1]
    y_data = to_categorical(labels, num_classes=len(actions))

    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, shuffle=True)

    model = Sequential([
        LSTM(64, input_shape=x_train.shape[1:3], activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(actions), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=200,
        callbacks=[
            ModelCheckpoint('models/model_webcam.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=30, verbose=1, mode='auto')
        ]
    )
    model.evaluate(x_test, y_test)

    predicted_y = model.predict(x_test)
    print('real Y - predicted Y')
    for idx, predict in enumerate(predicted_y):
        print(f'{y_test[idx]} - {predict}')

    with open('models/trainHistoryDict_webcam', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi, protocol=pickle.HIGHEST_PROTOCOL)


def train_radar():
    data = np.zeros((1, 50, 128))
    for idx, action in enumerate(actions):
        f_r = open(f'dataset/{action}_r/last_idx.txt', 'r')
        action_count = int(f_r.readline())
        f_r.close()

        for i in range(action_count):
            seq_data = np.load(f'dataset/{action}_r/{action}_{i + 1}.npy')
            if data.any() == 0:
                data = seq_data.reshape((1, 50, 128))
                data = np.append(data, np.full((1, 50, 1), idx), axis=2)
            else:
                data_ = seq_data.reshape((1, 50, 128))
                data_ = np.append(data_, np.full((1, 50, 1), idx), axis=2)
                data = np.append(data, data_, axis=0)

    indices = np.where(data == -math.inf)
    data[indices] = 0
    x_data = data[:, :, :-1]
    labels = data[:, 0, -1]
    y_data = to_categorical(labels, num_classes=len(actions))

    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, shuffle=True)

    model = Sequential([
        # LSTM(64, input_shape=x_train.shape[1:3], activation='relu'),
        # Dense(32, activation='relu'),
        # Dense(len(actions), activation='softmax')
        Convolution2D(32,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      activation='relu',
                      input_shape=(50, 128, 1)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Convolution2D(64,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(500, activation='relu'),
        Dense(len(actions), activation='softmax')
    ])

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=200,
        callbacks=[
            ModelCheckpoint('models/model_radar_cnn_test.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=50, verbose=1, mode='auto')
        ]
    )
    model.evaluate(x_test, y_test)

    predicted_y = model.predict(x_test)
    print('real Y - predicted Y')
    for idx, predict in enumerate(predicted_y):
        print(f'{y_test[idx]} - {predict}')

    with open('models/trainHistoryDict_radar_cnn_test', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi, protocol=pickle.HIGHEST_PROTOCOL)

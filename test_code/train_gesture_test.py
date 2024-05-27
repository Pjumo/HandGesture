import numpy as np
import os

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
actions = ['nothing', 'click', 'doubleclick']

data = np.zeros((1, 50, 100))
for action in actions:
    for i in range(80):
        seq_data = np.load(f'dataset/{action}/seq_{i+1}.npy')
        if data.any() == 0:
            data = seq_data.reshape((1, 50, 100))
        else:
            data = np.append(data, seq_data.reshape((1, 50, 100)), axis=0)

x_data = data[:, :, :-1]
labels = data[:, 0, -1]
y_data = to_categorical(labels, num_classes=len(actions))

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=100,
    callbacks=[
        ModelCheckpoint('../models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)

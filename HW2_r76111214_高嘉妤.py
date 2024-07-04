#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD


# 讀入 MNSIT 數據集
from tensorflow.keras.datasets import mnist

# mnist的load_data()會回傳已經先分割好的training data 和 testing data

# training data 總共有60000張圖片
# testing data 總共有10000張圖片
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 資料處理 

# reshape(28,28) -> 只有灰階（28,28,1個channel）
# /255:將每個 pixel 的值從 Int 轉成 floating point 同時做normalize
x_train = x_train.reshape(60000, 28, 28, 1) / 255
x_test = x_test.reshape(10000, 28, 28, 1) / 255

y_train = to_categorical(y_train, 10)  # (y_train, 10個種類)
y_test = to_categorical(y_test, 10)


# CNN
model = Sequential()

# HiddenLayer1
# Convolutional layer1
model.add(Conv2D(16, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))    # filters=16, size:(3*3), activationfunction=ReLu
# Pooling layer1
model.add(MaxPooling2D(pool_size=(2, 2)))   # size(2*2)

# HiddenLayer2
# Convolutional layer2
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))   # filters=32, size:(3*3), activationfunction=ReLu
# Pooling layer2
model.add(MaxPooling2D(pool_size=(2,2)))   # size(2*2)

# HiddenLayer3
# Convolutional layer3
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))  # filters=64, size:(3*3), activationfunction=ReLu
# Pooling layer3
model.add(MaxPooling2D(pool_size=(2, 2)))   # size(2*2)

# Flatten
model.add(Flatten())

# Fully Connected
model.add(Dense(64, activation='relu'))

# Output
model.add(Dense(10, activation='softmax'))

model.summary()


# Train 
# loss function=均方誤差(MSE), learning_rate=0.087
model.compile(loss='mse', optimizer=SGD(learning_rate=0.087), metrics=['accuracy'])

# Fit(batch_size=128, epochs=12)
history = model.fit(x_train, y_train, batch_size=128, epochs=15)

# Plot the loss curves
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.show()

# Plot the accuracy curves
plt.plot(history.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Training accuracy')
plt.show()

# Evaluation(Test)
loss, acc = model.evaluate(x_test, y_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix

# 記錄預測出來的值（0為在第0個位置,1為在第1個位置...）
Y_pred = model.predict(x_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
# 正確的值
Y_true = np.argmax(y_test, axis=1)

# compute confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
print(confusion_mtx)

# Plot
plt.imshow(confusion_mtx, interpolation='nearest')
plt.colorbar()
number_of_class=len(np.unique(Y_true))
tick_marks = np.arange(len(np.unique(Y_true)))
plt.xticks(tick_marks, range(number_of_class))
plt.yticks(tick_marks, range(number_of_class))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()


# save model
model.save('MnistModel')






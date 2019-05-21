import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

NAME = "Image-64"
pickle_X = open("X.pickle", "rb")
X = pickle.load(pickle_X)

pickle_y = open("y.pickle", "rb")
y = pickle.load(pickle_y)

X = X/255.0

#Design the network
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(3))
model.add(Activation('softmax'))

#Generate tensorboard logs for learning rate and accuracy etc.
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss='sparse_categorical_crossentropy', \
              optimizer='rmsprop', \
              metrics=['accuracy'])

model.fit(X, y,
          batch_size=32,
          epochs=20,
          validation_split=0.3,
          callbacks=[tensorboard])

#Save the model
model.save('cd_trained.model')

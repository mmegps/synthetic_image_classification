import tensorflow as tf
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.keras.layers import Conv2D, MaxPooling2D

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
#from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

NAME = "CNN-64"

'''
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss='binary_crossentropy', \
              optimizer='adam', \
              metrics=['accuracy'])

model.fit(X, y,
          batch_size=32,
          epochs=8,
          validation_split=0.3,
          callbacks=[tensorboard])

'''

DATADIR = "./test_images"
CATEGORIES = ["c0", "c1", "c2"]

test_data = []
IMG_SIZE = 64

def create_test_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([new_array, class_num])
            except Exception as e:
                pass

create_test_data()
print (len(test_data))

import random
random.shuffle(test_data)

#for sample in test_data[:10]:
#    print (sample[1])

#Create Model
X_test = []
y_test = []

for features,label in test_data:
    X_test.append(features)
    y_test.append(label)

#print (X_test[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#X_test = X_test/255.0

#Load the model
model = tf.keras.models.load_model('cd_trained.model')
predictions = model.predict(X_test)

#print ("model.output_shape", model.output_shape)
print ("X_test.shape", X_test.shape)
#print ("X_test[0].shape", X_test.shape)

print (y_test)

#Lets see the images and their predictions
new_array = cv2.resize(X_test[0], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print (CATEGORIES[int(predictions[0][0])])

new_array = cv2.resize(X_test[1], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print (CATEGORIES[int(predictions[1][0])])

new_array = cv2.resize(X_test[2], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print (CATEGORIES[int(predictions[2][0])])

new_array = cv2.resize(X_test[3], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print (CATEGORIES[int(predictions[3][0])])

new_array = cv2.resize(X_test[4], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print (CATEGORIES[int(predictions[4][0])])

new_array = cv2.resize(X_test[5], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print (CATEGORIES[int(predictions[5][0])])

new_array = cv2.resize(X_test[6], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print (CATEGORIES[int(predictions[6][0])])

new_array = cv2.resize(X_test[7], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print (CATEGORIES[int(predictions[7][0])])

new_array = cv2.resize(X_test[8], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print (CATEGORIES[int(predictions[8][0])])

new_array = cv2.resize(X_test[9], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print (CATEGORIES[int(predictions[9][0])])

new_array = cv2.resize(X_test[10], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print (CATEGORIES[int(predictions[10][0])])

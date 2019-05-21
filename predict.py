import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle
import time

NAME = "CNN-64"
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

random.shuffle(test_data)

#Create Model
X_test = []
y_test = []

for features,label in test_data:
    X_test.append(features)
    y_test.append(label)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#Load the trained model
model = tf.keras.models.load_model('cd_trained.model')
predictions = model.predict(X_test)

#print ("model.output_shape", model.output_shape)
print ("X_test.shape", X_test.shape)
#print ("X_test[0].shape", X_test.shape)

#Test images and their predictions
new_array = cv2.resize(X_test[0], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[0])
print ("Prediction:", int(predictions[0][0]),int(predictions[0][1]),int(predictions[0][2]))

new_array = cv2.resize(X_test[1], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[1])
print ("Prediction:", int(predictions[1][0]),int(predictions[1][1]),int(predictions[1][2]))

new_array = cv2.resize(X_test[2], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[2])
print ("Prediction:", int(predictions[2][0]),int(predictions[2][1]),int(predictions[2][2]))

new_array = cv2.resize(X_test[3], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[3])
print ("Prediction:", int(predictions[3][0]),int(predictions[3][1]),int(predictions[3][2]))

new_array = cv2.resize(X_test[4], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[4])
print ("Prediction:", int(predictions[4][0]),int(predictions[4][1]),int(predictions[4][2]))

new_array = cv2.resize(X_test[5], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[5])
print ("Prediction:", int(predictions[5][0]),int(predictions[5][1]),int(predictions[5][2]))

new_array = cv2.resize(X_test[6], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[6])
print ("Prediction:", int(predictions[6][0]),int(predictions[6][1]),int(predictions[6][2]))

new_array = cv2.resize(X_test[7], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[7])
print ("Prediction:", int(predictions[7][0]),int(predictions[7][1]),int(predictions[7][2]))

new_array = cv2.resize(X_test[8], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[8])
print ("Prediction:", int(predictions[8][0]),int(predictions[8][1]),int(predictions[8][2]))

new_array = cv2.resize(X_test[9], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[9])
print ("Prediction:", int(predictions[9][0]),int(predictions[9][1]),int(predictions[9][2]))

new_array = cv2.resize(X_test[10], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[10])
print ("Prediction:", int(predictions[10][0]),int(predictions[10][1]),int(predictions[10][2]))

new_array = cv2.resize(X_test[11], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[11])
print ("Prediction:", int(predictions[11][0]),int(predictions[11][1]),int(predictions[11][2]))

new_array = cv2.resize(X_test[12], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[12])
print ("Prediction:", int(predictions[12][0]),int(predictions[12][1]),int(predictions[12][2]))

new_array = cv2.resize(X_test[13], (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap=plt.cm.binary)
plt.show()
print ("Actual:", y_test[13])
print ("Prediction:", int(predictions[13][0]),int(predictions[13][1]),int(predictions[13][2]))
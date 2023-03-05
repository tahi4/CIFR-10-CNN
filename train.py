import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets, layers, models


# DATA PREPERATION
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images/ 255

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]


# this for loop doesnt contribute to the model, mainly for our visual rep, to see whether the images are being labelled correctly
for i in range(16):
    plt.subplot(4,4,i+1) #4x4 grid, with each iteration we're choosing a placein grid to put imgs
    plt.xticks([]) #empty so no coordinatate system
    plt.yticks([]) #empty so no coordinatate system
    plt.imshow(training_images[i], cmap=plt.cm.binary) #show first 16 imgs #cm = column map
    plt.xlabel(class_name[training_labels[i][0]]) #beneath each image is the name

plt.show()


# NEURAL NETWORK

model = models.Sequential()
# convolutional layers basically filter for features in img
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3))) #input layer #32 neurons 3x3 convolution matrix#input shape is 32x32px 3 color channels
model.add(layers.MaxPooling2D(2,2)) #after a convolutional layer we always have a max pooling 2D layer to simplify result and reduces to essential info
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten()) #makes it a straight line
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  #output #10 cassifications

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))


loss, accuracy = model.evaluate(testing_images, testing_labels)

model.save('img_classifier.model')







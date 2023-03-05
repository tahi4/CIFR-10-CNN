import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets, layers, models


# DATA PREPERATION
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images/ 255

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]

model = models.load_model('img_classifier.model')

img = cv.imread('car.jpg') #default is BGR but data was train on RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img])/255) #bcz all our training data was divided by 255
index = np.argmax(prediction) #most likely point 
print(f"I think it is a {class_name[index]}")

plt.show()


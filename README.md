# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

![image](https://user-images.githubusercontent.com/63336975/190658976-d67ea3b9-3950-45b8-bacd-8b7d99170bd2.png)

## Neural Network Model

Include the neural network model diagram.
![nn_model](https://user-images.githubusercontent.com/63336975/190659080-f5b03d0c-b735-4b71-aff9-21bc5c693283.png)


## DESIGN STEPS

### STEP 1:
Download and load the dataset

### STEP 2:
Scale the dataset between it's min and max values

### STEP 3:
Using one hot encode, encode the categorical values

### STEP 4:
Split the data into train and test

### STEP 5:
Build the convolutional neural network model

### STEP 6:
Train the model with the training data

### STEP 7:
Plot the performance plot

### STEP 8:
Evaluate the model with the testing data

## PROGRAM
``` python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(110, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
img = image.load_img('/content/sample_data/test_sample.png')
type(img)
img = image.load_img('/content/sample_data/test_sample.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/63336975/190659719-ed700f7b-8f7d-4ef7-885b-5bae4553b653.png)


### Classification Report

![image](https://user-images.githubusercontent.com/63336975/190670524-a186d4dd-49e4-4736-88fd-ddaf95a727aa.png)


### Confusion Matrix

![image](https://user-images.githubusercontent.com/63336975/190670646-d9301c66-4cba-4183-a94f-fb36ea173c62.png)


### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/63336975/190670970-5acafcdf-de22-4397-984c-3dadcf1121dd.png)

## RESULT
Successfully developed a convolutional deep neural network for digit classification and verified the response for scanned handwritten images.

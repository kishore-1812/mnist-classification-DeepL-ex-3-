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

https://github.com/kishore-1812/mnist-classification-DeepL-ex-3-/blob/main/Copy_of_Ex03_minist_classification.ipynb

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

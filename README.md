# CodeClause_project1_Hand-Written-Digit-Recognization
AI Project on Hand Written Digit Recognization

"""
Project for Handwritten Digit Recognition using machine learning with CNN

Created on Mon Jul 10 11:13:14 2023

Project Name : Handwritten Digit Recognition using Machine Learning
Author : Ms. Debalina Dasgupta
         UEMK
Program Description :
---------------------
Handwritten digit recognition is the process to provide the ability to machines 
to recognize human handwritten digits. The current project takes advantage of 
neural network for detecting such handwritten digits.

The program uses Convolutional Neural Networks (CNN), a special type of Deep
Learning Algorithm that take the image as an input and learn the various 
features of the image through filters. For this purpose the program uses the
MNIST (Modified National Institute of Standards and Technology) dataset which 
is a widely used dataset of handwritten digits that contains 60,000 handwritten 
digits for training a machine learning model and 10,000 handwritten digits for 
testing the model.

At the beginning, the program loads the training data and the testing data 
including labels using Keras library. The dimension of the training data is 
(60000x28x28) which is reshaped to a matrix of shape (60000x28x28x1) to feed 
into the CNN model.

The program then creates the CNN model by adding convolutional layers with ReLu
functiom and Max pooling layers successively. Then the hidden layers with 
softmax activation function are added to the model. A dropout layer is also 
added for deactivating some of the neurons in order to prevent any over fitting 
of the model. Finally, the output layer of 10 classes (representing 10 digits
from 0 to 9) with softmax activation function is added to the CNN model. The 
model is then compiled using compile() method of keras with Adam optimizer. 
The CNN model is then trained by feeding the MINST taining data using the 
fit() function of Keras with epochs values 10 and batch size of 128. 

To evaluate the accuracy of the model, the MINST test dataset of 10,000 images
is used. The model was tried with different sizes of kernel, such as, 3x3 &
2x2 and different epochs values of 5, 10 & 20 and dropout values of 0.25, 0.4, 
0.6 & 0.8. It is found that an accuracy of 99.34% is obtained while using the 
3x3 kernel with an epochs value of 10, two Maxpool layers and two droupt layers 
with value 0.25 and 0.5 respectively. The model is saved in a file named as 
cnn_model_3x3.keras in the current working directory.

Finally, a GUI program using tkinter is written to hand draw the digits and 
detect the digits using the model.

Requirements :
--------------
The program requires Keras library and the Tkinter library (for GUI building) 
to be installed in the machine.

References :
------------
https://keras.io/api/models/
https://www.stanford.edu/search/?q=Convolutional+Neural+Networks&search_type=web&submit=
https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
http://deeplearning.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/

https://data-flair.training/blogs/convolutional-neural-networks-tutorial/
https://towardsdatascience.com/convolution-neural-network-for-image-processing-using-keras-dc3429056306
https://www.geeksforgeeks.org/python-tensorflow-tf-keras-layers-conv2d-function/

https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/
https://www.analyticsvidhya.com/blog/2021/11/newbies-deep-learning-project-to-recognize-handwritten-digit/
https://www.geeksforgeeks.org/handwritten-digit-recognition-using-neural-network/

# -*- coding: utf-8 -*-
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
(60000*28*28) which is reshaped to a matrix of shape (60000*28*28*1) to feed 
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

#--------- To check image from MNIST Test dataset
#x_test[0] represents digit 7
import numpy as np
img=x_test[0,:,:,0]
arr = img.reshape(1,28,28,1)
res = model.predict([arr])
np.argmax(res[0])

#--------- To check image from any image file
from PIL import Image
img = Image.open('.\\Digits\\9_gray.jpg')
img.show()
img_arr = np.array(img)
arr = img_arr.reshape(1,28,28,1)
arr = arr/255.0
res=model.predict([arr])
res[0]
np.argmax(res[0])

#--------- To check image from any image file in Google Colab
import numpy as np
img = Image.open('hwdr/Digits/0_gray.jpg')
img_arr = np.array(img)
arr = img_arr.reshape(1,28,28,1)
arr = arr/255.0
res=model.predict([arr])
res[0]
np.argmax(res[0])

"""
#---------- Accuracy using Models with 3x3 kernels
#  Kernel Activa/Epochs/Maxpool/Dropout  Test Loss/Accuracy    Working                        Model File Name
#  3x3    Adam     1     Once    0.6     0.04841028153896332                                  cnn_model_3x3_epochs_1_Adam.keras
#                                        0.9832000136375427
#  3x3    Adam     5     Once    0.6     0.026718106120824814 0,2,4(E),6,8                    cnn_model_3x3_epochs_5_Adam_DO_0.6_MP_1.keras
#                                        0.9914000034332275
#  3x3    Adam     10    Once    0.6     0.0312336552888155   0,2,4,5(E),6(E),7(EE),8         cnn_model_3x3_epochs_10_Adam_DO_0.6_MP_1.keras
#                                        0.9929999709129333
#  3x3    Adam     5     Once    0.8     0.025035323575139046 0,2,4,5(E),7(E),8               cnn_model_3x3_epochs_5_Adam_DO_0.8_MP_1.keras
#                                        0.9922999739646912
#  3x3    Adam     10    Once    0.8     0.027058420702815056 0,2,3(EE),4,5(E),6,7,8          cnn_model_3x3_epochs_10_Adam_DO_0.8_MP_1.keras
#                                        0.9919999837875366

#  3x3    Adam     5     Twice   0.6     0.02801498770713806  0,2,4,5(E),7(E),8               cnn_model_3x3_epochs_5_Adam_DO_0.6_MP_2.keras
#                                        0.9901999831199646
#  3x3    Adam     10    Twice   0.6     0.03041088581085205  0,1,2,4,5(E),6,7(E),8           cnn_model_3x3_epochs_10_Adam_DO_0.6_MP_2_New.keras
#                                        0.9909999966621399
#  3x3    Adam     5     Twice   0.8     0.02510056644678116  0,1(E),2,4,8                    cnn_model_3x3_epochs_5_Adam_DO_0.8_MP_2.keras
#                                        0.9916999936103821
#  3x3    Adam     10    Twice   0.8     0.021174518391489983 0,2(E),2,8                      cnn_model_3x3_epochs_10_Adam_DO_0.8_MP_2.keras
#                                        0.9926000237464905
#  3x3    Adam     5     Twice  0.25,0.5 0.02729630470275879  0,2,4,7,8                       cnn_model_3x3_epochs_5_Adam_DO_0.25_0.5_MP_2.keras
#                                        0.9904999732971191
#**3x3    Adam     10    Twice  0.25,0.5 0.020842351019382477 0,1,2,3(E),4,5,6,7,8,9(EE)      cnn_model_3x3_epochs_10_Adam_DO_0.25_0.5_MP_2.keras
#                                        0.9934999942779541
#  3x3    Adam     5     Twice  0.25,0.6 0.020867079496383667 0,2,4,6,7,8                     cnn_model_3x3_epochs_5_Adam_DO_0.25_0.6_MP_2.keras
#                                        0.9933000206947327
#**3x3    Adam     10    Twice  0.25,0.6 0.020470496267080307 0,1,2,3(E),4,5(E),6,7(E),8,9(EE)cnn_model_3x3_epochs_10_Adam_DO_0.25_0.6_MP_2.keras
#                                        0.9923999905586243

#---------- Accuracy using Models with 2x2 kernels
#  Kernel Activa/Epochs/Maxpool/Dropout  Test Loss/Accuracy    Working                        Model File Name
#  2x2    Adam     5     Once    0.4     0.03549906983971596   0,2,3(E),4,5(E),7(E),8         cnn_model_2x2_epochs_5_Adam_DO_0.4_MP_1.keras
#                                        0.9882000088691711
#  2x2    Adam     10    Once    0.4     0.044536709785461426  2,4,6,7(E),8                   cnn_model_2x2_epochs_10_Adam_DO_0.4_MP_1.keras
#                                        0.9884999990463257
#  2x2    Adam     5     Once    0.6     0.031121524050831795  0,2,4,5,6,7,8                  cnn_model_2x2_epochs_5_Adam_DO_0.6_MP_1.keras
#                                        0.9897000193595886
#**2x2    Adam     10    Once    0.6     0.04242975637316704   0,1,2,3(E),4,5,6,7,8           cnn_model_2x2_epochs_10_Adam_DO_0.6_MP_1.keras
#                                        0.9878000020980835
#  2x2    Adam     5     Once    0.8     0.030068401247262955  0,2,4,5,6,8                    cnn_model_2x2_epochs_5_Adam_DO_0.8_MP_1.keras
#                                        0.9904000163078308
#  2x2    Adam     10    Once    0.8     0.026376964524388313  0,2,4,5(E),6,7(E),8            cnn_model_2x2_epochs_10_Adam_DO_0.8_MP_1.keras
#                                        0.9911999702453613
#**2x2    Adam     5     Once   0.25,0.5 0.03371323272585869   0,1(E),2,3,4,5,6,7(E),8        cnn_model_2x2_epochs_5_Adam_DO_0.25_0.5_MP_1.keras
#                                        0.9884999990463257
#  2x2    Adam     10    Once   0.25,0.5 0.03580410033464432   0,1(E),2,3(E),4,6,8            cnn_model_2x2_epochs_10_Adam_DO_0.25_0.5_MP_1.keras
#                                        0.9904000163078308
#**2x2    Adam     5     Once   0.25,0.6 0.029691318050026894  0,1(E),2,3(E),4,5,6,7(E),8     cnn_model_2x2_epochs_5_Adam_DO_0.25_0.6_MP_1.keras
#                                        0.9904000163078308
#* 2x2    Adam     10    Once   0.25,0.6 0.03449367731809616   0,1(EE),2,3(E),4,5(E),6,7(E),8 cnn_model_2x2_epochs_10_Adam_DO_0.25_0.6_MP_1.keras
#                                        0.9908000230789185

#**2x2    Adam     5     Twice   0.4     0.036401767283678055  0,1(E),2,3,4,6,8               cnn_model_2x2_epochs_5_Adam_DO_0.4_MP_2.keras
#                                        0.9890999794006348
#**2x2    Adam     10    Twice   0.4     0.03366348147392273   0,1(E),2,3(E),4,5,6,7,8        cnn_model_2x2_epochs_10_Adam_DO_0.4_MP_2.keras
#                                        0.9904999732971191

#  2x2    Adam     5     Twice   0.6     0.0290076844394207    0,2,3,4,5,6,8                  cnn_model_2x2_epochs_5_Adam_DO_0.6_MP_2.keras
#                                        0.991100013256073
#  2x2    Adam     5     Twice   0.8     0.03265228495001793   0,1,2,3,4,5(E),8               cnn_model_2x2_epochs_5_Adam_DO_0.8_MP_2.keras
#                                        0.989300012588501
#  2x2    Adam     5     Twice  0.25,0.5 0.029492290690541267  0,2,4(Err),6,8                 cnn_model_2x2_epochs_5_Adam_DO_0.25_0.5_MP_2.keras
#                                        0.9896000027656555
#  2x2    Adam     10    Twice  0.25,0.5 0.03241463005542755   0,1(E),2,3(E),4,6,8            cnn_model_2x2_epochs_10_Adam_DO_0.25_0.5_MP_2.keras
#                                        0.9904999732971191


#---------- Import the libraries
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

#---------- Define global variables
model_no = 1
batch_size = 128
epochs = 10

#----------********** Load the MNIST dataset for training and testing
#Size of X_train = 60000 x 28 x 28
#Size of Y_train = 60000
(x_train,y_train), (x_test, y_test)= mnist.load_data()
print("******************** MNIST data loaded successfully.....")
print("Shape of Input Training Data(x) : ",x_train.shape)
print("Shape of Input Training Data(y) : ",y_train.shape)
print("Shape of Input Test Data(x)     : ",x_test.shape)
print("Shape of Input Test Data(y)     : ",y_test.shape)

#----------********** Preprocess the data
#---------- Reshape the data and Convert the data into float values
x_train=x_train.reshape(x_train.shape[0], 28,28,1).astype('float32')
x_test=x_test.reshape(x_test.shape[0], 28,28,1).astype('float32')
input_shape = (28, 28, 1)

#---------- Normalize the data and Convert class vectors to binary class matrices
x_train=x_train/255.0
x_test=x_test/255.0
y_train= keras.utils.to_categorical(y_train)
y_test= keras.utils.to_categorical(y_test)
num_classes=y_train.shape[1]

print("******************** Input data reshaped and normalized successfully.....")
print("Shape of Training Data(x) : ",x_train.shape)
print("Shape of Training Data(y) : ",y_train.shape)
print("Shape of Test Data(x)     : ",x_test.shape)
print("Shape of Test Data(y)     : ",y_test.shape)
print("Number of classes         : ",num_classes)
print("Input shape               : ",input_shape)

#----------********** Function for creating the CNN model
def model_1():
    global input_shape, num_classes, model_no, batch_size, epochs
    model=Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    #model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#----------********** Train the model
model = model_1()
hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("******************** The model has been trained successfully.....")
print("model_no   : ",model_no)
print("batch_size : ",batch_size)
print("epochs     : ",epochs)

#---------- Save the model in the file
#model.save("F:\\Debalina\\UEM\\Sem7\\Internship\\HWDR\\model.keras")
#model_file = ".\\CNN_Model_"+str(model_no)+".keras"
model_file = ".\\cnn_model_3x3.keras"
model.save(model_file)

print("******************** The model is saved in the file "+model_file+".....")

#----------********** Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("******************** The model has been evaluated successfully.....")
print('Test loss     :', score[0])
print('Test accuracy :', score[1])

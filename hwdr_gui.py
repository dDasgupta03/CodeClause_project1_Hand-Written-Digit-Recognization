# -*- coding: utf-8 -*-
"""
Program for drawing the digit and capturing the image for predicting the digit
using the model developed by machine learning algorithm

Created on Tue Jul 11 20:43:43 2023
author : Debalina Dasgupta

Program Description :
---------------------
This program uses tkinter library to develop a GUI interface for hand drawing 
the digit on a canvas. It then capture the image and predicts the digit using 
the CNN model.

Reference :
https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/

#---------- To check the last drawn image directly.
g_img.show()
g_img_red.show()
g_img_gray.show()
g_img_gray.save('g_img_gray.jpg')
g_img_red.save('g_img_red.jpg')

g_img_arr[:,13:16]
img_arr=g_img_arr/255.0
img_arr[:,13:16]
arr=img_arr.reshape(1,28,28,1)
model = load_model('cnn_model_1.keras')
res=model.predict([arr])
res[0]
np.argmax(res[0])

PIL.ImageGrab.grab() method takes a snapshot of the screen. The pixels inside 
the bounding box are returned as an “RGB” image on Windows or “RGBA” on macOS. 
If the bounding box is omitted, the entire screen is copied.
from PIL import Image, ImageGrab
im = ImageGrab.grab(bbox = (0,0,300,300))
im.show()

(x, y, windowWidth, windowHeight) = rect

"""

#---------- Import the libraries
from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np

#---------- Define global variables
#g_img = Image.open('.\\Digits\\1.jpg')
#g_img_red = g_img.resize((28,28))
#g_img_gray = g_img.convert('L')
#g_img_arr = np.array(g_img_gray)

#---------- Load the model developed by machine learning algorithm
#model = load_model('cnn_model_2x2_epochs_10_Adam_DO_0.6_MP_1.keras')
#model = load_model('cnn_model_3x3_epochs_10_Adam_DO_0.25_0.6_MP_2.keras')
model = load_model('cnn_model_3x3.keras')

#----------********** Function for predicting the digit drawn
def predict_digit(img):
    #global g_img, g_img_red, g_img_gray, g_img_arr
    #g_img = img
    #print(img.size)
    #---------- Resize the image to 28x28 pixels
    img = img.resize((28,28))
    #g_img_red = img
    #---------- Convert the image from rgb to grayscale and to numpy array
    img = img.convert('L')
    #g_img_gray = img
    img = np.array(img)
    #g_img_arr = img
    #---------- Reshape and normalize the array
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #---------- Predict the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

#----------********** Digit class for drawing and capturing image for prediction
class Digit(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        #---------- Create the elements - canvas, labels and buttons
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command = self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)

        #---------- Create the Grid structure and add the elements
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    #---------- Erase all from the dwawing area
    def clear_all(self):
        self.canvas.delete("all")

    #---------- Read the image from the dwawing area and predict the digit
    def classify_handwriting(self):
        #---------- Get the handle of the canvas
        HWND = self.canvas.winfo_id()
        #---------- Get the coordinate of the canvas
        rect = win32gui.GetWindowRect(HWND)
        (x,y,w,h) = rect
        im = ImageGrab.grab(bbox = (x+40,y+40,1.2*w,1.2*h))
        #im = ImageGrab.grab(rect)

        digit, acc = predict_digit(im)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')

    #---------- Draw the line in the the dwawing canvas follwoing the mouse button click
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

#----------********** Main Program
d = Digit()
mainloop()
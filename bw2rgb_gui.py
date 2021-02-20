#!/usr/bin/env python
# coding: utf-8

# BW2RGB / Multimedia Processing
# Gourav Siddhad
# 09-November-2019

# ----------------------------------------------------------------------------

from PIL import Image, ImageTk
from tkinter import filedialog, Canvas, PhotoImage
from tkinter.ttk import Progressbar
import tkinter as tk
from skimage.color import rgb2lab, lab2rgb
from sklearn.model_selection import train_test_split
from skimage.io import imsave
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import keras.backend as K
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten
from keras.layers import Input, Dense, Layer, Reshape, ReLU, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.models import Model, Sequential, model_from_json
from tensorflow import set_random_seed
from numpy.random import seed
import tensorflow as tf
import os
print('Importing Libraries', end='')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seed(1607)
set_random_seed(1607)


print(' - Done')

# ----------------------------------------------------------------------------

H, W, Ci, Co = 128, 128, 1, 2

json_file = open('bw2rgb.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('bw2rgb.h5')
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()

# ----------------------------------------------------------------------------


def convert_img512(img):
    if len(img.shape) == 3:
        a, b, c = img.shape
    else:
        a, b = img.shape

    if a > 512 or b > 512:
        if a >= b:
            scale_percent = 512/a
            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            img = cv2.resize(img, (width, height),
                             interpolation=cv2.INTER_AREA)
        else:
            scale_percent = 512/b
            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            img = cv2.resize(img, (width, height),
                             interpolation=cv2.INTER_AREA)

    if len(img.shape) == 3:
        a, b, c = img.shape
        nimg = np.zeros((512, 512, 3))
        adiff, bdiff = 512-a, 512-b
        nimg[adiff//2:a+adiff//2, bdiff//2:b+bdiff//2] = img
    else:
        a, b = img.shape
        nimg = np.zeros((512, 512))
        adiff, bdiff = 512-a, 512-b
        nimg[adiff//2:a+adiff//2, bdiff//2:b+bdiff//2] = img

    return np.array(nimg, dtype='uint8')

# ----------------------------------------------------------------------------


master = tk.Tk()
master.title('Gray to Color Conversion')
width, height = 1050, 600
xcord = master.winfo_screenwidth() // 2 - width // 2
ycord = master.winfo_screenheight() // 2 - height // 2
master.geometry("%dx%d+%d+%d" % (width, height-20, xcord, ycord-20))

# ----------------------------------------------------------------------------
# Third Column
thirdFrame = tk.Frame(master, padx=5, pady=5)
thirdFrame.grid(row=1, column=0, sticky='nsew')

# ----------------------------------------------------------------------------
# Fourth Column
fourthFrame = tk.Frame(master, padx=5, pady=5)
fourthFrame.grid(row=1, column=1, sticky='nsew')

# ----------------------------------------------------------------------------

path = tk.StringVar()
path.set('C:/Users/Gourav Siddhad/Documents/BW2RGB/Project/test_large/')


def select_file():
    global path
    path = filedialog.askopenfilename(initialdir="C:/Users/Gourav Siddhad/Documents/BW2RGB/Project/test_large/",
                                      title="Select Image",
                                      filetypes=(("JPEG files", "*.jpg"), ("all files", "*.*")))
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if len(img.shape) == 3:
        img_lab = rgb2lab(img)
        sample_img = 255*img_lab[:, :, 0]/100.0
        img_l = img_lab[:, :, 0]/100.0
    else:
        sample_img = img
        img_l = img/255.0

    # ---------------
    render = ImageTk.PhotoImage(Image.fromarray(convert_img512(sample_img)))
    img_label1 = tk.Label(thirdFrame, image=render)
    img_label1.image = render
    img_label1.grid(row=0, column=0)
    master.update_idletasks()
    # ---------------

    if float(img.shape[0]//H) == img.shape[0]/H:
        a = img.shape[0]
    else:
        a = H*(img.shape[0]//H) + H

    if float(img.shape[1]//W) == img.shape[1]/W:
        b = img.shape[1]
    else:
        b = W*(img.shape[1]//W) + W

    pimg = np.zeros((a, b))
    pimg[0:img.shape[0], 0:img.shape[1]] = img_l

    new_img = np.zeros((a, b, Ci+Co))
    total = (pimg.shape[0]//H)

    for j in range(pimg.shape[0]//H):
        for k in range(pimg.shape[1]//W):
            t_l = pimg[j*H:(j+1)*H, k*W:(k+1)*W]

            t_ab = model.predict(t_l.reshape(1, H, W, Ci))
            t_ab = t_ab.reshape(H, W, Co)

            temp = np.zeros((H, W, Ci+Co))
            temp[:, :, 0] = t_l*100.0
            temp[:, :, 1:] = t_ab*128.0
            temp = lab2rgb(temp)

            new_img[j*H:(j+1)*H, k*W:(k+1)*W, :] = temp.reshape(H, W, Ci+Co)
        progress['value'] = 100*j/total
        master.update_idletasks()
    progress['value'] = 100
    master.update_idletasks()

    newimg = np.array(new_img*255, dtype='uint8')
    newimg = newimg[0:img.shape[0], 0:img.shape[1]]

    cv2.imwrite('sample_out.jpg', cv2.cvtColor(newimg, cv2.COLOR_RGB2BGR))

    # ---------------
    render = ImageTk.PhotoImage(Image.fromarray(convert_img512(newimg)))
    img_label2 = tk.Label(fourthFrame, image=render)
    img_label2.image = render
    img_label2.grid(row=0, column=0)
    master.update_idletasks()
    # ---------------

    a, b, _ = np.array(newimg).shape
    if a > 512 and b > 512:
        cv2.imshow('RGB Output', cv2.cvtColor(newimg, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# First Column
firstFrame = tk.Frame(master, padx=10, pady=10)
firstFrame.grid(row=0, column=0, sticky='nsew')

button1 = tk.Button(firstFrame, text='Open Image', command=select_file)
button1.grid(row=0, column=0)

# ----------------------------------------------------------------------------
# Second Column
secondFrame = tk.Frame(master, padx=10, pady=10)
secondFrame.grid(row=0, column=1, sticky='nsew')

progress = Progressbar(secondFrame, orient=tk.HORIZONTAL,
                       length=200, mode='determinate')
progress.grid(row=0, column=0)

# ----------------------------------------------------------------------------
master.mainloop()

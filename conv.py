"""
CONVOLUTION OF IMAGE
"""

from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np
import os
def filterFeatures(arr):
    for index,color_rgb in enumerate(arr): arr[index] = (0,0,0) if color_rgb[0] != 30 else (30/255, 30/255, color_rgb[2]/255)
    return arr
def ConvFile(newfile,save):
    getDay = lambda str: str.split("\\")[2]
    img = Image.open(newfile)
    img_val = filterFeatures(list(img.getdata()))#current image data
    x = np.array(img_val).reshape(-1,img.height,img.width,3)
    x = tf.constant(x, dtype=tf.float32)
    kernel_in = np.array([
        [
            [[0,0,0],[0,0,0],[0,0,0]], [[1,1,1],[1,1,1],[1,1,1]], [[0,0,0],[0,0,0],[0,0,0]]
        ],
        [
            [[1,1,1],[1,1,1],[1,1,1]], [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]], [[1,1,1],[1,1,1],[1,1,1]]
        ],
        [
            [[0,0,0],[0,0,0],[0,0,0]], [[1,1,1],[1,1,1],[1,1,1]], [[0,0,0],[0,0,0],[0,0,0]]
        ]
    ])
    # full length -> kernal height
    # length in [height] -> kernal width
    # length in [height][width] -> input channels
    # length in [height][width][input] -> output channels
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    out = np.abs(np.array(tf.nn.conv2d(x, kernel, strides=[1, 3, 3, 1], padding='VALID'))[0])
    out *= 255/out.max()
    imgSave = Image.new("RGB",(out.shape[1], out.shape[0]),"black")
    imgSave_value = imgSave.load()
    for x in range(imgSave.width):
        for y in range(imgSave.height):
            imgSave_value[x,y] = (int(out[y][x][0]),int(out[y][x][1]),int(out[y][x][2]))
    imgSave.save(f"{save}{getDay(newfile)}")
    print("the image is of dimension(x,y)=>",imgSave.size,"and total pixels are =>",imgSave.size[0]*imgSave.size[1])
    img.close()
    imgSave.close()
ConvFile("F:\\imagesTrain\\0.png","")

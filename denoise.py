"""
DENOISE THE CONVOLUTED IMAGES
"""

from tensorflow import keras
import tensorflow as tf
from PIL import Image
import glob, random
import numpy as np

def createDataset(IMAGE_FILES1,TrainData): # GLOB PATHS
    glob_files = glob.glob(IMAGE_FILES1)
    pixelValue = list()
    for index in range(len(glob_files)):
        newfile = glob_files[index]
        imgX = Image.open(newfile)
        dataRawX = list(imgX.getdata())
        img_valX = [dataRawX[i][0]/255 for i in range(len(dataRawX)) ]
        pixelValue.append(img_valX)
        imgX.close()
    x,r = list(),list()
    #input and desired outputs
    for i in range(0,TrainData):
        rand = random.randint(0,len(pixelValue)-2)
        r.append(rand)
        x.append(pixelValue[rand])
        print("random values for training",rand)
    x = np.array(x)
    print(x.shape)
    return x,r
def Train(dataPath,TrainData,save):
    X,imagesTrainID = createDataset(dataPath,TrainData)
    Y = X.copy()
    print(Y.shape)
    model = keras.Sequential([
        keras.layers.Input(shape=(7200)),
        # keras.layers.BatchNormalization(),
        keras.layers.Dense(units=1000, activation='relu'),
        keras.layers.Dense(units=7200, activation='sigmoid')
    ])
    #spec of model
    model.compile(optimizer='adam', loss='mse')
    #fit
    model.fit(x = X, y = Y, epochs = 100, steps_per_epoch = 25,validation_split=0.1)
    #saves the ANN
    model.save(f'{save}.h5')

Train("F:\\Convolution\\*.png",2500,"Models/RainFall_Denoise")

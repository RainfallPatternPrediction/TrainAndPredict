"""
PREDICTS FROM THE CONVOLUTED IMAGE
"""

from tensorflow import keras
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import glob, random
import numpy as np

def createDataset(IMAGE_FILES1,TrainData): # GLOB PATHS AS THE 2 ARGUMENTS
    glob_files = glob.glob(IMAGE_FILES1)
    pixelValue = list()
    for index in range(len(glob_files)):
        newfile = glob_files[index]
        imgX = Image.open(newfile)
        dataRawX = list(imgX.getdata())
        img_valX = [dataRawX[i][0]/255 for i in range(len(dataRawX)) ]
        pixelValue.append(img_valX)
        imgX.close()
    x,y,r = list(),list(),list()
    #input and desired outputs
    for i in range(0,TrainData):
        rand = random.randint(0,len(pixelValue)-2)
        r.append(rand)
        x.append(pixelValue[rand])
        y.append(pixelValue[rand+1])
        print("random values for training",rand)
    x = np.array(x).reshape(-1,7200,)
    y = np.array(y).reshape(-1,7200,)
    return x,y,r
def Train(dataPath,TrainData,save):
    X,y,imagesTrainID = createDataset(dataPath,TrainData)

    model = keras.models.load_model("Models/RainFall_Predictor.h5")
    # model = keras.Sequential([
    #     keras.layers.Input(shape=(7200,)),
    #     keras.layers.Dense(units=5000, activation='relu'),
    #     keras.layers.Dropout(0.1),
    #     keras.layers.Dense(units=7200, activation='sigmoid')
    # ])
    # spec of model
    model.compile(optimizer='adam', loss="mse")
    model.summary()
    #fit
    model.fit(X, y, epochs = 4,steps_per_epoch = 50,batch_size=BATCH,verbose=1)
    #saves the ANN
    model.save(f'{save}.h5')

Train("F:\Convolution",5000,"Models/RainFall_Predictor")

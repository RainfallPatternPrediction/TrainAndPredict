"""
DECODER FOR THE CONVOLUTED IMAGES
"""
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import glob, random
import numpy as np
def createDataset(IMAGE_FILES1,IMAGE_FILES2,TrainData): # GLOB PATHS AS THE 2 ARGUMENTS
    def filterFeatures(arr):
        for index,color_rgb in enumerate(arr): arr[index] = 0 if color_rgb[0] != 30 else color_rgb[2]/255
        return arr
    glob_files_input = glob.glob(IMAGE_FILES1)
    glob_files_output = glob.glob(IMAGE_FILES2)
    pixelValueInput = list()
    pixelValueOutput = list()
    for index in range(len(glob_files_input)):
        newfile_input = glob_files_input[index]
        newfile_output = glob_files_output[index]
        imgX = Image.open(newfile_input)
        imgY = Image.open(newfile_output)

        # <difference>
        dataRawX = list(imgX.getdata())
        img_valX = [dataRawX[i][0]/255 for i in range(len(dataRawX)) ]
        img_valY = filterFeatures(list(imgY.getdata()))
        # </difference>

        pixelValueInput.append(img_valX)
        pixelValueOutput.append(img_valY)
        imgX.close()
        imgY.close()
    x,y,r = list(),list(),list()
    #input and desired outputs
    for i in range(0,TrainData):
        rand = random.randint(0,len(pixelValueInput)-2)
        r.append(rand)
        x.append(pixelValueInput[rand])
        y.append(pixelValueOutput[rand])
        print("random values for training",rand)
    # <difference>
    x = np.array(x).reshape(-1,7200,)
    y = np.array(y).reshape(-1,64800,)
    # </difference>
    return x,y,r

def Train(Inputs,Outputs,TrainData,save):
    X,y,imagesTrainID = createDataset(Inputs,Outputs,TrainData)
    print("Image numbers",np.array(imagesTrainID))
    print("got all image Data")
    model = keras.models.load_model("Models/RainFall_Decoder.h5")
    # model = keras.Sequential([
    #     keras.layers.Input(shape=(7200,)),
    #     keras.layers.Dense(units=5000, activation='relu'),
    #     keras.layers.Dropout(0.2),
    #     keras.layers.Dense(units=64800, activation='sigmoid')
    # ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x = X, y = y, epochs = 100, steps_per_epoch = 25)
    model.save(f'{save}.h5')

Train("F:\\Convolution\\*.png","F:\\imagesTrain\\*.png",2500,"Models/RainFall_Decoder")

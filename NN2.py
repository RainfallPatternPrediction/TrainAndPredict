from tensorflow import keras
import tensorflow as tf
from PIL import Image
import glob, random
import numpy as np
TrainData = 10000 #total training data
#data parsing
def createDataset(IMAGE_FILES):# glob file
    def filterFeatures(arr):
        for index,color_rgb in enumerate(arr):
            arr[index] = 0 if color_rgb[0] != 30 else color_rgb[2]/255
        return arr
    glob_files = glob.glob(IMAGE_FILES)
    #converting to black and white
    pixelValue = list()
    for index in range(len(glob_files)):
        newfile = glob_files[index]
        img = Image.open(newfile)
        img_val = filterFeatures(list(img.getdata()))#current image data
        pixelValue.append(img_val)#containes data for all images
        img.close()
    x,y = [],[]
    #input and desired outputs
    for i in range(0,TrainData):
        rand = random.randint(0,len(pixelValue)-2)
        x.append(pixelValue[rand])
        y.append(pixelValue[rand+1])
    x = np.array(x).reshape(-1,64800,)
    y = np.array(y).reshape(-1,64800,)
    return x,y
X,y = createDataset("F:\\imagesTrain\\*.png")#change for your location of the parsed images
print("got all image Data")
#Shape of the model
model = keras.Sequential([
    keras.layers.Input(shape=(64800,)),
    keras.layers.Dense(units=10000, activation='relu'),
    keras.layers.Dense(units=64800, activation='sigmoid')
])
#spec of model
model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
#fit
history = model.fit(
    x = X,
    y = y,
    epochs = 100,
    steps_per_epoch = 100
)
#saves the ANN
model.save('RainFall_Prediction.h5')

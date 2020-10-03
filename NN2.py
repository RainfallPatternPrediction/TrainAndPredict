from tensorflow import keras
import tensorflow as tf
from PIL import Image
import glob, random
import numpy as np
TrainData = 10000 #total training data
#data parsing
def createDataset(IMAGE_FILES):# glob file
    def flat(arr):
        a = []
        parse = lambda x: x/255
        for i in arr:
            a.append(parse(i.flatten()))
        return np.array(a)
    glob_files = glob.glob(IMAGE_FILES)
    #converting to black and white
    pixelValue = list()
    for index in range(len(glob_files)):
        newfile = glob_files[index]
        img = Image.open(newfile)
        img = img.convert('L')
        pixelValue.append(list(img.getdata()))#containes data for all images
        img.close()
    X,y = [],[]
    #input and desired outputs
    for i in range(0,TrainData):
        rand = random.randint(0,len(pixelValue)-2)
        X.append(pixelValue[rand])
        y.append(pixelValue[rand+1])
    X = flat(np.array(X).reshape(-1,180,360))
    y = flat(np.array(y).reshape(-1,180,360))
    return X,y
X,y = createDataset("F:\\imagesTrain\\*.png")#change for your location of the parsed images
print("got all image Data")
# #Shape of the model
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

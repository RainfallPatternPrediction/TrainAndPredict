"""
USING THE ENCODER-DECODER

(in progress)
"""

from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np

def createDataset(IMAGE_FILES):# glob file
    def flat(arr):
        a = []
        parse = lambda x: x/255
        for i in arr:
            a.append(parse(i.flatten()))
        return np.array(a)
    img = Image.open(IMAGE_FILES).convert('L')
    pixelValue = list(img.getdata()) #containes data for all images
    img.close()
    return flat(np.array(pixelValue).reshape(-1,60,120))

def createImagePredictor(val,date):
    img = Image.new("RGB",(120,60),"black")
    img_value = img.load()
    for y in range(60):
        for x in range(120):
            color = int(val[y][x]*255)
            img_value[x,y] = (color,color,color)
    img.save(f'ImageConvolution/predicted-encoded-{date}.png')
    img.close()
def createImageDenoise(val,date):
    img = Image.new("RGB",(120,60),"black")
    img_value = img.load()
    for y in range(60):
        for x in range(120):
            color = int(val[y][x]*255)
            img_value[x,y] = (color,color,color)
    img.save(f'ImageDenoised/predicted-encoded-{date}.png')
    img.close()
def createImageDecoder(val,date):
    img = Image.open('world.bmp')
    img_value = img.load()
    for y in range(180):
        for x in range(360):
            blue = int(val[y][x]*255)
            if(blue > 50):
                img_value[x,y] = (30,30,blue)
    img.save(f'ImageConverted/predicted-decoded-{date}.png')
    img.close()

# TODO: FIX THIS FOR TENSOR INPUTS
def AI(image,options):
    X = createDataset(image)
    model1 = keras.models.load_model(options["model"])
    futureDataEncoded = model1.predict(x = X)
    if options["clear"]==True: keras.backend.clear_session()
    futureDataEncoded = futureDataEncoded.reshape(-1,options["shape"][0],options["shape"][1]).tolist()[0]
    if options["model"].split("_")[1].split(".")[0] == 'Decoder':
        createImageDecoder(futureDataEncoded,options["date"])
    elif options["model"].split("_")[1].split(".")[0] == 'Denoise':
        createImageDenoise(futureDataEncoded,options["date"])
    else:
        createImagePredictor(futureDataEncoded,options["date"])

def Test(start,numberDays,startImg):
    path = startImg
    for Date in range(start,numberDays):
        AI(path,{
                "date":Date+1,
                "model":"Models/RainFall_Predictor.h5",
                "shape":(60,120),
                "clear":True
                }) # encode predict
        AI(f'ImageConvolution/predicted-encoded-{Date+1}.png',{
                                                "date":Date+1,
                                                "model":"Models/RainFall_Denoise.h5",
                                                "shape":(60,120),
                                                "clear":True
                                                }) # denoise
        path = f'ImageDenoised/predicted-encoded-{Date+1}.png'

# Test(0,3,"F:\\Convolution\\0.png")# 'ImageConvolution/predicted-encoded-5.png'

model1 = keras.models.load_model("Models/RainFall_Decoder.h5")
for Date in range(0,3):
    X = createDataset(f'ImageDenoised/predicted-encoded-{Date+1}.png')
    futureDataEncoded = model1.predict(x = X)
    futureDataEncoded = futureDataEncoded.reshape(-1,180,360).tolist()[0]
    createImageDecoder(futureDataEncoded,Date+1)


# Date = 0
# AI(f"F:\\Convolution\\{Date}.png",{"date":Date+1,"model":"RainFall_Predictor.h5","shape":(60,120)}) # encode predict
# AI(f"predicted-encoded-{Date+1}.png",{"date":Date+1,"model":"RainFall_Denoise.h5","shape":(60,120)}) # denoise
# AI(f"predicted-encoded-{Date+1}.png",{"date":Date+1,"model":"RainFall_Decoder.h5","shape":(180,360)}) # decoding of prediction
#
# AI(f"F:\\Convolution\\{Date+1}.png",{"date":Date+2,"model":"RainFall_Predictor.h5","shape":(60,120)}) # encode predict
# AI(f"predicted-encoded-{Date+2}.png",{"date":Date+2,"model":"RainFall_Denoise.h5","shape":(60,120)}) # denoise
# AI(f"predicted-encoded-{Date+2}.png",{"date":Date+2,"model":"RainFall_Decoder.h5","shape":(180,360)}) # decoding of prediction

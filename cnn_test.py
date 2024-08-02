#Loading the packages
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np 

#Loading the model
model = tensorflow.keras.models.load_model('keras_model.h5')

data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)

def predict(img):
    #Loading the image
    image = Image.open(img)
    size = (224,224)
    image = ImageOps.fit(image,size,Image.ANTIALIAS)

    #turn the image into array
    image_array = np.asarray(image)

    #Normalizing the image 
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array 

    #Prediction
    prediction = model.predict(data)
    print(prediction)

    idx = np.argmax(prediction)
    print(idx)

    if(idx == 0):
        print("No  Accident Detected")
    else:
        print("Accident Detected")

i = 1
while(i):
    img = input('Enter the name of image for testing: ')
    predict(img)
    i = int(input('Enter 1 to continue and 0 to exit: '))
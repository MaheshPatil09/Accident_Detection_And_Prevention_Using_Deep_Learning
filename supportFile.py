import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from twilio.rest import Client
import cv2


account_sid = 'ACc58c08f54bb4bd1705c361dc3e4fa216'
auth_token = 'f3ee4dc8fe2dbe0497e0457a204dce70'
client = Client(account_sid, auth_token)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def predict():
    # Replace this with the path to your image
    image = Image.open('static/images/test_image.jpg')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    idx = np.argmax(prediction)
    #print(num)

    if idx == 0:
        '''
        message = client.messages \
        .create(
                body = "Accident Detected",
                from_='+17073556535',
                to="+918459552560"
            )

        print('SMS Sent')
        '''
        return "Accident Detected"

    elif idx == 1:
        return "No Accident Detected"
    else:
        return "Unknown Vehicle"



def get_frame():
	webcam = cv2.VideoCapture(0)
	while True:
		ret, frame = webcam.read()
		if ret == False:
			break
		cv2.imwrite('static/images/test_image.jpg',frame)
		result = predict()
		cv2.putText(frame, result, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))

		#talk(result)                                    #speech output

		imgencode=cv2.imencode('.jpg',frame)[1]
		stringData=imgencode.tostring()

		yield (b'--frame\r\n'
			b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

	webcam.release()
	cv2.destroyAllWindows()
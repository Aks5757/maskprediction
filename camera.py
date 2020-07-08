import cv2
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

mask_model = tensorflow.keras.models.load_model('keras_model.h5')
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

labels = ['Mask ON','NO Mask']
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(-1)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        frame: object
        ret, frame = self.video.read()
        print("Type of Video is ",self.video)
        pilimag1 = frame.copy()
        data1 = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        PIL_image = Image.fromarray(np.uint8(pilimag1)).convert('RGB')
        size = (224, 224)
        image = ImageOps.fit(PIL_image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data1[0] = normalized_image_array
        prediction = mask_model.predict(data1)

        print(prediction)
        label = labels[prediction.argmax()]
        #frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,
         #                  interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in face_rects:
            if label == 'NO Mask':
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if label == 'Mask ON':
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            break
        # DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

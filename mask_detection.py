
from flask import Flask, request, jsonify, render_template

from flask import redirect, url_for
import os
from werkzeug.utils import secure_filename

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
mask_model = tensorflow.keras.models.load_model('keras_model.h5')

#masmodel = load_model('keras_model.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/model', methods=['POST'])
def model():
    # Main page
    ml_models = [str(x) for x in request.form.values()]
    print("ml_model--->>>",ml_models)
    if ml_models[0]=='mask':
        return redirect(url_for("mask_detect"))

@app.route('/mask_detect',methods=['GET'])
def mask_detect():
    # Image2Text page
    return render_template('mask.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("filepath ->>>>",file_path)
        # Make prediction

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = Image.open(file_path)

        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = np.asarray(image)


        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = mask_model.predict(data)
        print(prediction)


        if prediction[0][0] > prediction[0][1]:
            maskdata ='Mask On'
        else:
            maskdata = 'No Mask'
        return maskdata
    return None


if __name__ == "__main__":
    app.run(debug=True)

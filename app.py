import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

model = load_model('model/face_mask_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_mask(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    input_pred_label = np.argmax(prediction)
 # Get the prediction scores
    if input_pred_label >= 0.9:# Assuming a threshold of 0.5 for mask detection
        result = "Mask Detected"
    else:
        result = "No Mask Detected"

    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            result = detect_mask(filename, model)
            return render_template("index.html", filename=file.filename, result=result)

    return render_template("index.html", filename=None, result=None)

if __name__ == '__main__':
    app.run(debug=True)

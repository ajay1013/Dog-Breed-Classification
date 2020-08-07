# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:29:33 2020

@author: Ajay Kumar
"""

from flask import Flask, request
import tensorflow
import numpy as np
import pickle
from flasgger import Swagger
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
Swagger(app)

dict=open("dicts.pkl", "rb")
classifier=pickle.load(dict)

model = load_model('model_vgg19.h5')

@app.route("/")
def welcome():
    return  "welcome everyone"


@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's Authenticate the DOG BREED
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    
    img = image.load_img('val/PNEUMONIA/person1946_bacteria_4874.jpeg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    classes = model.predict(img_data)
    prediction = classifier[classes]
    return "This is a dog of " + str(prediction)


if __name__=="__main__":
    app.run()

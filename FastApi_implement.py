import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image, ImageOps 
from io import BytesIO
import sys
import base64

# Create the FastAPI app
app = FastAPI()

# Load the model from the specified path
def get_model(path: str):
    return load_model(path)

# Load the MNIST model

model_path = "/Users/anikbhowmick/Python/Big_Data_Assignment/A06/MNIST_model.keras"
model = get_model(model_path)
model.trainable=False

# Function to preprocess image and make prediction

def predict_digit(model, data_point):
    pred = model.predict(data_point)
    prediction = tf.argmax(pred,axis=-1)
    c_score = np.max(pred)
    return str(prediction[0].numpy()),str(c_score)

# API endpoint to accept image upload and return prediction
def format_image(image):
    """
    get a pillow image
    """
    return image.resize((28,28))


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    accepted_formats = ['.jpeg', '.jpg', '.png']
    file_format = os.path.splitext(file.filename)[1].lower()
    if file_format not in accepted_formats:
        raise HTTPException(status_code=400, detail="Bad file format. Accepted formats are .jpeg, .jpg, .png")
    file_name = os.path.splitext(file.filename)[0]
    image = Image.open(BytesIO(content))
    #convert the image to gray scale first
    image = image.convert('L')
    if image.size!=(28,28):
        image = format_image(image)
    flat = np.array(image).reshape(-1)/255.0
    flat = flat[None,:]
    output, c_score = predict_digit(model, flat)
    return {
        "actual":file_name,
        "predicted": output,
        "confidence":c_score}



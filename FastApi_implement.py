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
#load the pretrained model
model = get_model(model_path)
# set the model in inference mode
model.trainable=False

# Function to preprocess image and make prediction

def predict_digit(model, data_point):

    # get the prediction containg the score 
    pred = model.predict(data_point)
    # get the class label
    prediction = tf.argmax(pred,axis=-1) 
    c_score = np.max(pred)# store the confidence score
    return str(prediction[0].numpy()),str(c_score)

# API endpoint to accept image upload and return prediction
def format_image(image):
    """
    get a pillow image
    """
    # resize the image in 28X28 format
    return image.resize((28,28))


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # load the image in the byte format
    content = await file.read()
    accepted_formats = ['.jpeg', '.jpg', '.png']
    file_format = os.path.splitext(file.filename)[1].lower()
    # check for the image file is valid or not
    if file_format not in accepted_formats:
        # raise the error message if the file is wrong in format
        raise HTTPException(status_code=400, detail="Bad file format. Accepted formats are .jpeg, .jpg, .png")
    file_name = os.path.splitext(file.filename)[0]
    image = Image.open(BytesIO(content))
    #convert the image to gray scale first
    image = image.convert('L')
    if image.size!=(28,28):
        # if the image is not 28 by 28 resize it 
        image = format_image(image)
    flat = np.array(image,dtype='float32').reshape(-1)/255.0# flatten the image and normalize in 0 to 1 scale
    flat = flat[None,:]
    output, c_score = predict_digit(model, flat)
    return {
        "actual":file_name,
        "predicted": output,
        "confidence":c_score}



# Model deployment with FastAPI
As part of MLOps, this assignment aims to deploy the ML model on a web server using Fast Api. FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints. For keeping the task simple a simple MNIST digit classifier is chosen to serve for this purpose. The API is used in the Swagger UI for checking the model prediction. We will briefly first go through problem statement
## Problem Statement
* Create a function “def load_model(path:str) -> Sequential” which will load the model saved at the supply path on the disk and return the keras.src.engine.sequential.Sequential model.
* Create a function “def predict_digit(model:Sequential, data_point:list) -> str” that will take the image serialized as an array of 784 elements and returns the predicted digit as string.
* Create an API endpoint “@app post(‘/predict’)” that will read the bytes from the uploaded image to create an serialized array of 784 elements. The array shall be sent to ‘predict_digit’ function to get the digit. The API endpoint should return {“digit”:digit”} back to the client.
* Test the API via the Swagger UI or Postman, where you will upload the digit as an image (28x28 size).
* Create a new function “def format_image” which will resize any uploaded images to a 28x28 grey scale image followed by creating a serialized array of 784 elements.
* Now, draw an image of a digit yourself using tools such as “ms paint” or equivalent using your touch screen or the mouse pointer. Upload your hand drawn image to your API and find out if your API is able to figure out the digit correctly. Repeat this exercise for 10 such drawings and report the performance of your API/model combo.
## Descriptions of folders

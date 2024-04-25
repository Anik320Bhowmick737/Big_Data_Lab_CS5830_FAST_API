# Model deployment with FastAPI
As part of MLOps, this assignment aims to deploy the ML model on a web server using Fast Api. FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints. For keeping the task simple a simple MNIST digit classifier is chosen to serve for this purpose. The API is used in the Swagger UI for checking the model prediction. We will briefly first go through problem statement
## Problem Statement
* Create a function “def load_model(path:str) -> Sequential” which will load the model saved at the supply path on the disk and return the keras.src.engine.sequential.Sequential model.
* Create a function “def predict_digit(model:Sequential, data_point:list) -> str” that will take the image serialized as an array of 784 elements and returns the predicted digit as string.
* Create an API endpoint “@app post(‘/predict’)” that will read the bytes from the uploaded image to create an serialized array of 784 elements. The array shall be sent to ‘predict_digit’ function to get the digit. The API endpoint should return {“digit”:digit”} back to the client.
* Test the API via the Swagger UI or Postman, where you will upload the digit as an image (28x28 size).
* Create a new function “def format_image” which will resize any uploaded images to a 28x28 grey scale image followed by creating a serialized array of 784 elements.
* Now, draw an image of a digit yourself using tools such as “ms paint” or equivalent using your touch screen or the mouse pointer. Upload your hand drawn image to your API and find out if your API is able to figure out the digit correctly. Repeat this exercise for 10 such drawings and report the performance of your API/model combo.
## Descriptions of folders and files
1. `handwritten images` contains image files which were written manually and some MNIST images for model testing
2. `results` contains the prediction pictures from Swagger UI
3. `FastApi_implement.py` contains main code file for our task
4. `MNIST_Model.py` is the codefile for training the model
5. `requirements.txt` contains all the necessary libraries for our task
6. `MNIST_Model.keras` is the keras trained model on MNIST digit data
## Model architecture
The model is a simple ANN model with 2 hidden layers and 1 output layer with 10 nodes for 10 digits from 0 to 9. The Model architecture can be  summarized below 


<img width="264" alt="architecture" src="https://github.com/Anik320Bhowmick737/Big_Data_Lab_CS5830_FAST_API/assets/97800241/025288a0-b77e-42fb-9326-00b53e0a4b00">

## Implementation code 
For loading the model following code is used:
```ruby
def get_model(path: str):
    return load_model(path)

# Load the MNIST model

model_path = "/Users/anikbhowmick/Python/Big_Data_Assignment/A06/MNIST_model.keras"
#load the pretrained model
model = get_model(model_path)
# set the model in inference mode
model.trainable=False
```
Because our UI expects image to be uploaded there we used post routing in the FastAPI. The predict function is wrapped in the post routing. So this code will run whenever we enter into the web UI post routing. 
```ruby
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
```
for prediction of the image following code is used. It returns the predicted class and it's confidence score.
```ruby
def predict_digit(model, data_point):

    # get the prediction containg the score 
    pred = model.predict(data_point)
    # get the class label
    prediction = tf.argmax(pred,axis=-1) 
    c_score = np.max(pred)# store the confidence score
    return str(prediction[0].numpy()),str(c_score)
```
## Running the FastAPI
After everything is set up in the terminal we have to write
```ruby
uvicorn FastApi_implement:app --reload
```
It will launch the local host on default http://127.0.0.1:8000 port. For running this API in the webserver we used Swagger UI just use http://127.0.0.1:8000/docs



import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense, Input, RandomRotation
from keras.models import Sequential
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.losses import SparseCategoricalCrossentropy
import numpy as np
# for training the model with the best hyperparameters of previous assignment 
def get_architecture():
    # sequentail definition of the model
    model = Sequential(
[
    Input(shape=(784,)),
    Dense(units = 256, activation = 'sigmoid'),
    Dense(units = 128, activation = 'sigmoid'),
    Dense(units = 10, activation='softmax')
])
    return model

def train_model():
    model = get_architecture()
    model.compile(loss=SparseCategoricalCrossentropy(),metrics=['acc'])
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # make sure the images are in white backgrround instead of black background it will make the inference easier for hand written digits for testing
    x_train=255-x_train
    x_test=255-x_test
    # flatten the datapoints to 784 dimensional vector as the model is simple ANN and expects linear data points
    x_train = np.array(x_train,dtype='float32').reshape(60000,-1)/255.0
    x_test = np.array(x_test,dtype='float32').reshape(10000,-1)/255.0

    model.fit(x_train,y_train, validation_split = 0.2,epochs =10)
    model.evaluate(x_test,y_test)
    # save the model for use in FAST API later
    model.save('MNIST_model.keras')

if __name__=="__main__":
    train_model()
    # start the code from here


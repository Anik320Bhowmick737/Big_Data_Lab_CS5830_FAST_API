import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense, Input, RandomRotation
from keras.models import Sequential
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.losses import SparseCategoricalCrossentropy
import numpy as np

def get_architecture():
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
    
    #x_train = RandomRotation(0.2,fill_mode='nearest',interpolation='nearest')(x_train)
    #x_test = RandomRotation(0.2,fill_mode='nearest',interpolation='nearest')(x_test)
    x_train=255-x_train
    x_test=255-x_test
    x_train = np.array(x_train).reshape(60000,-1)/255.0
    x_test = np.array(x_test).reshape(10000,-1)/255.0

    model.fit(x_train,y_train, validation_split = 0.2,epochs =10)
    model.evaluate(x_test,y_test)
    model.save('MNIST_model.keras')

if __name__=="__main__":
    train_model()


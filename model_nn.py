import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.mnist

(train_x,train_y),(test_x,test_y) = data.load_data()

train_x = (train_x/255.0)
test_x = (test_x/255.0)


train_x = train_x.reshape(60000, 28, 28, 1)
test_x = test_x.reshape(10000, 28, 28, 1)

print(train_x.shape)
print(train_y.shape)

model = keras.Sequential([
    keras.layers.Conv2D(16,(3,3),activation='relu',input_shape = (28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(16,(3,3),activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation = "relu"),
    keras.layers.Dense(10,activation = "softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(train_x,train_y,epochs=10)

results = model.evaluate(test_x, test_y)
print(results)

model.save("model_mnist_num.h5")


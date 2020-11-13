import tensorflow as tf
import numpy as np
from tensorflow import keras

def house_model(y_new):

    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float) # Your Code Here#
    ys = np.array([1,1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)# Your Code Here#
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) # Your Code Here#
    model.compile(optimizer='sgd', loss='mean_squared_error')# Your Code Here#
    model.fit(xs, ys, epochs=1000)# Your Code here#
    return model.predict(y_new)[0]

    prediction = house_model([7.0])
print(prediction)

import numpy as np
import keras
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras import initializers
from keras import optimizers
from keras.utils import plot_model
from keras.layers import *
from load_data import load_dataset
import pandas as pd

train_x, train_y, test_x, test_y, n_classes, genre = load_dataset(verbose=1, mode="Train", datasetSize=1)
# datasetSize = 0.75, this returns 3/4th of the dataset.

# Expand the dimensions of the image to have a channel dimension. (nx128x128) ==> (nx128x128x1)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

# Normalize the matrices.
train_x = train_x / 255.
test_x = test_x / 255.
#n_classes=8

model = Sequential()
#model.add(Conv2D(filters=128, kernel_size=[10,10], kernel_initializer = initializers.he_normal(seed=1), activation="relu", input_shape=(128,128,1)))
model.add(Conv1D(filters=256, kernel_size=4, kernel_initializer = initializers.he_normal(seed=1), activation="relu", input_shape=(128,128,1)))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=2, strides=2))


model.add(Conv1D(filters=256, kernel_size=4, kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2))

model.add(Conv1D(filters=512, kernel_size=4, kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2))


model.add(GlobalAveragePooling2D())
#model.add(BatchNormalization())
model.add(Dense(1024, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
#model.add(BatchNormalization())
model.add(Dense(n_classes, activation="softmax", kernel_initializer=initializers.he_normal(seed=1)))


print(model.summary())


model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
pd.DataFrame(model.fit(train_x, train_y, epochs=32, verbose=1, validation_split=0.1).history).to_csv("Saved_Model/training_history.csv")
score = model.evaluate(test_x, test_y, verbose=1)

model.save("Saved_Model/Model.h5")

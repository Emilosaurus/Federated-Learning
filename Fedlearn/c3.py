import flwr as fl
import tensorflow as tf

from tensorflow import keras

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Define Flower client
class CifarClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return model.get_weights()

  def fit(self, parameters, config):
    model.set_weights(parameters)
    model.fit(x_train, y_train, epochs=1, batch_size=32)
    return model.get_weights(), len(x_train), {}

  def evaluate(self, parameters, config):
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())

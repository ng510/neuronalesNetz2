# Einfach Neuronales Netz, welches auf den FashionMNIST Daten trainiert wird
# Vorhersage, zu welcher Kategorie das Bild gehört
# FashionMNIST: https://github.com/zalandoresearch/fashion-mnist

import gzip
import numpy as np

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

def open_images(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16)            .reshape(-1, 28, 28)            .astype(np.float32)

def open_labels(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)
   
    
X_train = open_images("../data/fashion/train-images-idx3-ubyte.gz")
y_train = open_labels("../data/fashion/train-labels-idx1-ubyte.gz")

X_test = open_images("../data/fashion/t10k-images-idx3-ubyte.gz")
y_test = open_labels("../data/fashion/t10k-labels-idx1-ubyte.gz")

# Kategorisiere labels für die 10 Ausgänge
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model mit 2 Hidden Layern und den 10 Ausgängen für die 10 Kategorien
model = Sequential()

model.add(Dense(200, activation="sigmoid", input_shape=(784,)))
model.add(Dense(100, activation="sigmoid", input_shape=(784,)))
model.add(Dense(10, activation="softmax"))

# mögliche optimizer: rmsprop, adam, sgd
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    X_train.reshape(60000, 784),
    y_train,
    epochs=50,
    batch_size=1000)

#Evaluiere Genauigkeit des Models auf Train und Test Daten, um Model möglicherweise anzupassen
#Ausgabe: ['loss', 'acc'] => Genauigkeit in Prozent im 2. Eintrag 
print(model.evaluate(X_train.reshape(-1, 784), y_train))

print(model.evaluate(X_test.reshape(-1, 784), y_test))

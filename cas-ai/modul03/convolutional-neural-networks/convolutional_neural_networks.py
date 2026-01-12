# ==========================================
# Neural Network – MNIST (handgeschriebene Ziffern)
# Dataset Source: https://www.tensorflow.org/datasets/catalog/mnist
# ==========================================

# Bibliotheken importieren
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

###
# Getting started:
# pip install tensorflow keras numpy matplotlib
###

# MNIST Datensatz laden
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Die Bilddaten umformen und normalisieren
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Modell erstellen
model = models.Sequential()

# Faltungsschicht (Convolutional Layer)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Max-Pooling Schicht
model.add(layers.MaxPooling2D((2, 2)))

# Weitere Faltungsschicht
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Noch eine Max-Pooling Schicht
model.add(layers.MaxPooling2D((2, 2)))

# Flatten Schicht, um das Ergebnis in ein 1D-Array zu verwandeln
model.add(layers.Flatten())

# Dense Schicht (Fully connected)
model.add(layers.Dense(64, activation='relu'))

# Ausgabeschicht (10 Klassen für die Ziffern 0 bis 9)
model.add(layers.Dense(10, activation='softmax'))

# Modell kompilieren
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Modell trainieren
model.fit(x_train, y_train, epochs=5, batch_size=64)

# Modell evaluieren
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Beispielbild aus dem Testdatensatz anzeigen
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f'Label: {y_test[0]}')
plt.show()

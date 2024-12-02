import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
from src.read import load_train_data, load_evaluate_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import HeNormal
import csv
import os

file_path = "resultados_simple.csv"
file_exists = os.path.isfile(file_path)

epochs_g = 20
batch_size_g=32
learning_rate_g = 0.001
activation_g = 'softmax'
kernel_initializer_g = 'glorot_uniform'
patience_g = 3
early_stop = 'yes'

train_images, train_labels = load_train_data()
test_images, test_labels = load_evaluate_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

model = Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    Flatten(),
    Dense(10, activation=activation_g)
])

model.compile(optimizer=Adam(learning_rate=learning_rate_g),
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

 
early_stopping = EarlyStopping(monitor='val_loss', patience=patience_g)

model.fit(train_images, train_labels, 
          epochs=epochs_g, 
          batch_size=batch_size_g, 
          validation_data=(test_images, test_labels),
          callbacks=[early_stopping])

loss_test, accuracy_test = model.evaluate(test_images, test_labels)
loss_train, accuracy_train = model.evaluate(train_images, train_labels)

error_test = 100 - (accuracy_test * 100)
error_train = 100 - (accuracy_train * 100)

with open(file_path, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=';')
    if not file_exists:
        writer.writerow(["epochs", "batch_size", "learning_rate", "loss_train", "accuracy_train", "error_train", "loss_test", "accuracy_test", "error_test", "activation", "kernel_initializer", "patience", "early_stop"])
    writer.writerow([epochs_g, batch_size_g, learning_rate_g, loss_train, accuracy_train * 100, error_train, loss_test, accuracy_test * 100, error_test, activation_g, kernel_initializer_g, patience_g, early_stop])

model.save('model_simple_2.h5')

print(f"Tasa de error en el conjunto de prueba: {error_test:.2f}%")
print(f"Tasa de error en el conjunto de entrenamiento: {error_train:.2f}%")
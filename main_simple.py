import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
from src.read import load_train_data, load_evaluate_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
import numpy as np
from openpyxl import Workbook

# Hiperparámetros
epochs_g = 20
batch_size_g = 32
learning_rate_g = 0.001
activation_g = 'softmax'
patience_g = 3

# Cargar datos
train_images, train_labels = load_train_data()
test_images, test_labels = load_evaluate_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Dividir el conjunto de entrenamiento en 80% para entrenamiento y 20% para validación
validation_percentage = 0.2
indices = np.arange(train_images.shape[0])
np.random.shuffle(indices)
validation_size = int(validation_percentage * train_images.shape[0])

val_indices = indices[:validation_size]
train_indices = indices[validation_size:]

train_images_split = train_images[train_indices]
train_labels_split = train_labels[train_indices]

val_images_split = train_images[val_indices]
val_labels_split = train_labels[val_indices]

class TestSetEvaluationCallback(Callback):
    def __init__(self, test_images, test_labels, file_path):
        self.test_images = test_images
        self.test_labels = test_labels
        self.file_path = file_path
        self.epoch_data = []

    def on_epoch_end(self, epoch, logs=None):
        loss_test, accuracy_test = self.model.evaluate(self.test_images, self.test_labels, verbose=0)
        self.epoch_data.append((epoch + 1, accuracy_test, loss_test))

    def on_train_end(self, logs=None):
        wb = Workbook()
        ws = wb.active
        ws.title = "Resultados"
        ws.append(["Época", "Accuracy", "Loss"])
        for data in self.epoch_data:
            ws.append(data)
        wb.save(self.file_path)

model = Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    Flatten(),
    Dense(10, activation=activation_g, kernel_initializer='glorot_uniform')
])

model.compile(optimizer=Adam(learning_rate=learning_rate_g),
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=patience_g)
test_eval_callback = TestSetEvaluationCallback(test_images, test_labels, "grafica_simple.xlsx")

model.fit(train_images_split, train_labels_split, 
          epochs=epochs_g, 
          batch_size=batch_size_g, 
          validation_data=(val_images_split, val_labels_split),
          callbacks=[early_stopping, test_eval_callback])

loss_test, accuracy_test = model.evaluate(test_images, test_labels)
loss_train, accuracy_train = model.evaluate(train_images_split, train_labels_split)

error_test = 100 - (accuracy_test * 100)
error_train = 100 - (accuracy_train * 100)

print(f"Tasa de error en el conjunto de prueba: {error_test:.2f}%")
print(f"Tasa de error en el conjunto de entrenamiento: {error_train:.2f}%")

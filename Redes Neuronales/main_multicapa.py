import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
from src.read import load_train_data, load_evaluate_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from openpyxl import Workbook
import numpy as np

# Hiperparámetros
epochs_g = 20
batch_size_g = 32
learning_rate_g = 0.001
activation_g = 'sigmoid'
kernel_initializer_logic_units_g = 'lecun_uniform'
kernel_initializer_exit_g = 'glorot_uniform'
patience_g = 5

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

model = Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    Flatten(),
    Dense(256, activation=activation_g, kernel_initializer=kernel_initializer_logic_units_g),
    Dense(10, activation='softmax', kernel_initializer=kernel_initializer_exit_g)
])

model.compile(optimizer=Adam(learning_rate=learning_rate_g),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=patience_g)

model.fit(train_images_split, train_labels_split,
          epochs=epochs_g,
          batch_size=batch_size_g,
          validation_data=(val_images_split, val_labels_split),
          callbacks=[early_stopping])

loss_test, accuracy_test = model.evaluate(test_images, test_labels)
loss_train, accuracy_train = model.evaluate(train_images, train_labels)

error_test = 100 - (accuracy_test * 100)
error_train = 100 - (accuracy_train * 100)

print(f"Tasa de error en el conjunto de prueba: {error_test:.2f}%")
print(f"Tasa de error en el conjunto de entrenamiento: {error_train:.2f}%")

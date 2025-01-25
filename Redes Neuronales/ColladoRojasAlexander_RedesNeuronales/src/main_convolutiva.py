import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from read import load_train_data, load_evaluate_data
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import csv
import os
import time

#Parámetros e Hiperparámetros
epochs_g = 50
batch_size_g = 32
learning_rate_g = 0.01
momentum_g = 0.95
patience_g = 8

#Carga del conjunto de prueba y entrenamiento
train_images, train_labels = load_train_data()
test_images, test_labels = load_evaluate_data()

#Normalización y Reestructuración de las Imágenes
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

#80% Entrenamiento y 20% Validación
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

#Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=7,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

datagen.fit(train_images_split)

#Topología
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(10, activation='softmax', kernel_initializer='glorot_uniform')
])

#Configuración
optimizer = SGD(learning_rate=learning_rate_g, momentum=momentum_g)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=patience_g, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2, min_lr=1e-6)

#Medir tiempo
#El que tengo de 0.43% lo envié en el formulario (Uso la GPU)
start_time = time.time()

#Entrenamiento
model.fit(datagen.flow(train_images_split, train_labels_split, batch_size=batch_size_g),
          epochs=epochs_g,
          validation_data=(val_images_split, val_labels_split),
          callbacks=[early_stopping, reduce_lr],
          steps_per_epoch=len(train_images_split) // batch_size_g)

end_time = time.time()
training_time = end_time - start_time

print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")

#Evaluación
loss_test, accuracy_test = model.evaluate(test_images, test_labels)
loss_train, accuracy_train = model.evaluate(train_images_split, train_labels_split)

#Obtener medida tasa de errores
error_test = 100 - (accuracy_test * 100)
error_train = 100 - (accuracy_train * 100)

#Gugardar modelo
model.save('../modelo/modelo_entrenado.h5') #Este modelo cada vez que se ejecute se sobreescribe

print(f"Tasa de error en el conjunto de prueba: {error_test:.2f}%")
print(f"Tasa de error en el conjunto de entrenamiento: {error_train:.2f}%")

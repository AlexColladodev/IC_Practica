import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from src.read import load_train_data, load_evaluate_data
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import csv
import os

file_path = "resultados_cnn.csv"
file_exists = os.path.isfile(file_path)

epochs_g = 50
batch_size_g = 32
learning_rate_g = 0.0075 #0.01 o 0.005
momentum_g = 0.85 #0.9 o 0.95
patience_g = 8 # 7

train_images, train_labels = load_train_data()
test_images, test_labels = load_evaluate_data()

train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

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
    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),
    Dense(10, activation='softmax', kernel_initializer='glorot_uniform')
])

optimizer = SGD(learning_rate=learning_rate_g, momentum=momentum_g)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#val_loss o val_accuracy
early_stopping = EarlyStopping(monitor='val_loss', patience=patience_g, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, min_lr=1e-6) # 3 - 0.5 o 0.75

model.fit(train_images_split, train_labels_split,
          epochs=epochs_g,
          batch_size=batch_size_g,
          validation_data=(val_images_split, val_labels_split),
          callbacks=[early_stopping, reduce_lr])

loss_test, accuracy_test = model.evaluate(test_images, test_labels)
loss_train, accuracy_train = model.evaluate(train_images_split, train_labels_split)

error_test = 100 - (accuracy_test * 100)
error_train = 100 - (accuracy_train * 100)

#Debo arreglar esto
with open(file_path, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=';')
    if not file_exists:
        writer.writerow(["epochs", "batch_size", "learning_rate", "momentum", "loss_train", "accuracy_train", "error_train", "loss_test", "accuracy_test", "error_test", "optimizer"])
    writer.writerow([epochs_g, batch_size_g, learning_rate_g, momentum_g, loss_train, accuracy_train * 100, error_train, loss_test, accuracy_test * 100, error_test, "SGD"])

model.save('model_convolutiva_2.h5')

print(f"Tasa de error en el conjunto de prueba: {error_test:.2f}%")
print(f"Tasa de error en el conjunto de entrenamiento: {error_train:.2f}%")

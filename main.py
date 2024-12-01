import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
from src.read import load_train_data, load_evaluate_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

train_images, train_labels = load_train_data()
test_images, test_labels = load_evaluate_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(10, activation='softmax')
])

model.compile(optimizer="adam", 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

model.fit(train_images, train_labels, 
          epochs=20, 
          batch_size=32, 
          validation_data=(test_images, test_labels),
          callbacks=[early_stopping])

accuracy_test = model.evaluate(test_images, test_labels)
accuracy_train = model.evaluate(train_images, train_labels)
print(f"Accuracy en el conjunto de prueba: {accuracy_test:.4f}")
print(f"Accuracy en el conjunto de entrenamiento: {accuracy_train:.4f}")

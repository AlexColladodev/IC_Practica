import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from read import load_train_data, load_evaluate_data
import numpy as np
import os

# Cargar datos
train_images, train_labels = load_train_data()
test_images, test_labels = load_evaluate_data()

train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

model_path = '../modelo/modelo_entrenado_43.h5' #Este es el modelo que tuvo una tasa de error de 0.43%
if os.path.exists(model_path):
    print(f"Cargando modelo desde {model_path}...")
    model = load_model(model_path)
else:
    print(f"El modelo {model_path} no existe. Por favor, entrena y guarda un modelo primero.")
    exit()

# Evaluar en conjunto de prueba
loss_test, accuracy_test = model.evaluate(test_images, test_labels)
error_test = (1 - accuracy_test) * 100

# Evaluar en conjunto de entrenamiento
loss_train, accuracy_train = model.evaluate(train_images, train_labels)
error_train = (1 - accuracy_train) * 100

print(f"Porcentaje de accuracy en el conjunto de prueba: {accuracy_test:.4f}%")
print(f"Porcentaje de error en el conjunto de prueba: {error_test:.2f}%")
print(f"Porcentaje de accuracy en el conjunto de entrenamiento: {accuracy_train:.4f}%")
print(f"Porcentaje de error en el conjunto de entrenamiento: {error_train:.2f}%")

# Predicciones y guardar resultados de la cadena
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

output_file = "../modelo/cadena_etiquetas_modelo_43.txt"
with open(output_file, mode='w', encoding='utf-8') as f:
    for pred in predicted_classes:
        f.write(f"{pred}")

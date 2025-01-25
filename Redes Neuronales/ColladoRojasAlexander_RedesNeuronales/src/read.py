import numpy as np
import struct
import os

def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "El archivo de imágenes MNIST no tiene el formato esperado."
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, "El archivo de etiquetas MNIST no tiene el formato esperado."
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def load_train_data():
    train_images = load_mnist_images("../data/train-images.idx3-ubyte")
    train_labels = load_mnist_labels("../data/train-labels.idx1-ubyte")
    return train_images, train_labels

def load_evaluate_data():
    evaluate_images = load_mnist_images("../data/t10k-images.idx3-ubyte")
    evaluate_labels = load_mnist_labels("../data/t10k-labels.idx1-ubyte")
    return evaluate_images, evaluate_labels

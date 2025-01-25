import numpy as np
import random

def calcular_suma_fila(matriz):
    suma = np.sum(matriz, axis=1)
    return suma

def ordenar_descendente(vector):
    return np.argsort(-vector)

def ordenar_ascendente(vector):
    return np.argsort(vector)

def buscar_valor_vector(valor, vector, usado):
    for i, v in enumerate(vector):
        if v == valor and not usado[i]:
            usado[i] = True
            return i
    return -1

def greedy(tamanio_matriz, matriz_flujo, matriz_distancia):

    suma_flujo = calcular_suma_fila(matriz_flujo)
    suma_distancia = calcular_suma_fila(matriz_distancia)

    indices_flujo = ordenar_descendente(suma_flujo)
    indices_distancia = ordenar_ascendente(suma_distancia)

    usado_flujo = np.zeros(tamanio_matriz, dtype=bool)
    usado_distancia = np.zeros(tamanio_matriz, dtype=bool)
    solucion = np.empty(tamanio_matriz, dtype=int)

    for i in range(tamanio_matriz):
        flujo_idx = buscar_valor_vector(suma_flujo[indices_flujo[i]], suma_flujo, usado_flujo)
        distancia_idx = buscar_valor_vector(suma_distancia[indices_distancia[i]], suma_distancia, usado_distancia)
        solucion[flujo_idx] = distancia_idx

    return solucion

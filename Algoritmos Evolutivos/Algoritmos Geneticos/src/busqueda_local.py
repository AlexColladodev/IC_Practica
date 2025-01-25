import numpy as np
import random

def factoriza_bl(pos1, pos2, vector_permutado, matriz_flujo, matriz_distancia):
    tamanio_vector = len(vector_permutado)
    coste = 0
    for i in range(tamanio_vector):
        if i != pos1 and i != pos2:
            coste += (
                matriz_flujo[pos1][i] * (
                    matriz_distancia[vector_permutado[pos2]][vector_permutado[i]] -
                    matriz_distancia[vector_permutado[pos1]][vector_permutado[i]]
                ) +
                matriz_flujo[pos2][i] * (
                    matriz_distancia[vector_permutado[pos1]][vector_permutado[i]] -
                    matriz_distancia[vector_permutado[pos2]][vector_permutado[i]]
                ) +
                matriz_flujo[i][pos1] * (
                    matriz_distancia[vector_permutado[i]][vector_permutado[pos2]] -
                    matriz_distancia[vector_permutado[i]][vector_permutado[pos1]]
                ) +
                matriz_flujo[i][pos2] * (
                    matriz_distancia[vector_permutado[i]][vector_permutado[pos1]] -
                    matriz_distancia[vector_permutado[i]][vector_permutado[pos2]]
                )
            )
    return coste

def aplicar_movimiento(i, j, vector_permutado, matriz_flujo, matriz_distancia):
    vector_permutado[i], vector_permutado[j] = vector_permutado[j], vector_permutado[i]

def calcular_coste_solucion(vector_permutado, matriz_flujo, matriz_distancia):
    tamanio_vector = len(vector_permutado)
    coste = 0
    for i in range(tamanio_vector):
        for j in range(tamanio_vector):
            coste += matriz_flujo[i][j] * matriz_distancia[vector_permutado[i]][vector_permutado[j]]
    return coste

def bl(vector, matriz_flujo, matriz_distancia, iteraciones_maximas):
    vector_permutado = np.array(vector)
    vector_binario = np.zeros(len(vector), dtype=bool)
    tamanio_vector = len(vector)
    
    mejora_entorno = True

    for _ in range(iteraciones_maximas):
        if not mejora_entorno:
            break

        mejora_entorno = False

        for i in range(tamanio_vector):
            if not vector_binario[i]:
                mejora_coste = False

                for j in range(tamanio_vector):
                    if factoriza_bl(i, j, vector_permutado, matriz_flujo, matriz_distancia) < 0:
                        aplicar_movimiento(i, j, vector_permutado, matriz_flujo, matriz_distancia)
                        vector_binario[i] = False
                        vector_binario[j] = False
                        mejora_coste = True
                        mejora_entorno = True
                        break

                if not mejora_coste:
                    vector_binario[i] = True

    return vector_permutado.tolist()

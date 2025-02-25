from reader import Reader
from greedy import greedy
from busqueda_local import bl
from genetico import agg_pmx
import numpy as np
import random
import time

def comprobar_permutacion():
    permutacion = [131, 160, 233, 183, 180, 193, 159, 187, 170, 174, 89, 237, 31, 222, 191, 195, 146, 200, 155, 242, 208, 93, 252, 227, 246, 163, 112, 86, 152, 178, 218, 7, 157, 215, 165, 150, 189, 239, 10, 229, 25, 29, 22, 33, 111, 235, 1, 225, 27, 54, 185, 133, 81, 37, 52, 63, 101, 14, 42, 116, 137, 59, 204, 39, 135, 18, 74, 83, 66, 240, 248, 120, 46, 140, 198, 103, 78, 57, 142, 44, 212, 35, 64, 129, 107, 98, 76, 69, 122, 72, 125, 4, 38, 94, 82, 65, 80, 43, 45, 58, 75, 34, 219, 51, 113, 92, 124, 134, 48, 108, 8, 77, 62, 50, 100, 68, 231, 36, 41, 139, 254, 56, 121, 104, 123, 73, 30, 105, 47, 244, 115, 97, 55, 13, 85, 99, 53, 151, 149, 114, 144, 2, 136, 127, 16, 249, 241, 130, 96, 147, 49, 141, 102, 24, 61, 138, 153, 109, 110, 126, 173, 156, 162, 143, 128, 3, 118, 67, 5, 168, 148, 71, 171, 247, 158, 17, 145, 161, 28, 194, 26, 32, 164, 179, 167, 84, 154, 87, 40, 188, 15, 181, 176, 192, 11, 184, 175, 210, 197, 182, 20, 166, 117, 201, 190, 199, 177, 207, 205, 223, 172, 209, 196, 228, 202, 169, 214, 238, 12, 186, 203, 221, 220, 6, 91, 224, 211, 216, 79, 213, 88, 206, 217, 60, 230, 234, 236, 9, 95, 119, 226, 243, 0, 19, 245, 21, 23, 232, 70, 255, 250, 251, 253, 106, 90, 132]

    return permutacion

def calcular_coste(matriz_flujo, matriz_distancia, individuo):
    n = len(individuo)
    coste = 0
    for i in range(n):
        for j in range(n):
            coste += matriz_flujo[i][j] * matriz_distancia[individuo[i]][individuo[j]]
    return coste

def main():

    inicio = time.time()
    reader = Reader()

    archivo = "tai256c.dat"
    reader.establecer_nombre_archivo(archivo)
    reader.leer_matrices()

    matriz_flujo = reader.get_matriz_flujo()
    matriz_distancia = reader.get_matriz_distancia()
    tamanio_matriz = reader.get_tamanio_matriz()

    solucion = comprobar_permutacion()

    coste = calcular_coste(matriz_flujo, matriz_distancia, solucion)

    print(coste)




if __name__ == "__main__":
    main()

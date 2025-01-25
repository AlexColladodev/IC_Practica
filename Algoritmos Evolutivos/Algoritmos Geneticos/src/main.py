from reader import Reader
from greedy import greedy
from busqueda_local import bl
from genetico import agg_pmx
import numpy as np
import random
import time

def calcular_coste(matriz_flujo, matriz_distancia, individuo):
    n = len(individuo)
    coste = 0
    for i in range(n):
        for j in range(n):
            coste += matriz_flujo[i][j] * matriz_distancia[individuo[i]][individuo[j]]
    return coste

def nota(cost):
    best_cost = 44759294
    return max((5 - (100 * ((cost - best_cost)/cost))), 0)

def generar_poblacion(solucion_bl, tamanio_poblacion, random_seed=None):
    poblacion = []
    random.seed(random_seed)

    for _ in range(tamanio_poblacion):
        permutacion = solucion_bl.copy()

        tamanio_modificacion = random.randint(1, len(solucion_bl) // 2)

        for _ in range(tamanio_modificacion):
            idx1, idx2 = random.sample(range(len(solucion_bl)), 2)
            permutacion[idx1], permutacion[idx2] = permutacion[idx2], permutacion[idx1]

        poblacion.append(permutacion)
    return poblacion


def main():
    inicio = time.time()
    reader = Reader()

    #Semilla Buena
    semilla_bl = 7
    random_seed = 7

    archivo = "tai256c.dat"
    reader.establecer_nombre_archivo(archivo)
    reader.leer_matrices()

    matriz_flujo = reader.get_matriz_flujo()
    matriz_distancia = reader.get_matriz_distancia()
    tamanio_matriz = reader.get_tamanio_matriz()

    tamanio_poblacion = 300
    iter_max = 2000

    inicio_greedy = time.time()
    solucion_greedy = greedy(tamanio_matriz, matriz_flujo, matriz_distancia)
    fin_greedy = time.time()

    inicio_bl = time.time()
    solucion_bl = bl(solucion_greedy, matriz_flujo, matriz_distancia, iter_max)
    fin_bl = time.time()

    inicio_obtener_poblacion = time.time()
    poblacion = generar_poblacion(solucion_bl, tamanio_poblacion, random_seed)
    fin_obtener_poblacion = time.time()

    inicio_agg_pmx = time.time()
    mejor_solucion = agg_pmx(poblacion, matriz_flujo, matriz_distancia, tamanio_matriz, iter_max, prob_cruce=0.7, prob_mutacion=0.01, semilla=random_seed)
    fin_agg_pmx = time.time()

    fin = time.time()

    tiempo_total = fin - inicio

    tiempo_greedy = fin_greedy - inicio_greedy

    tiempo_bl = fin_bl - inicio_bl

    tiempo_poblacion = fin_obtener_poblacion - inicio_obtener_poblacion

    tiempo_pmx = fin_agg_pmx - inicio_agg_pmx

    coste_mejor = calcular_coste(matriz_flujo, matriz_distancia, mejor_solucion)

    nota_x = nota(coste_mejor)

    print(f"Archivo: {archivo}")
    print(f"Mejor solución: {mejor_solucion}")
    print(f"Coste de la mejor solución: {coste_mejor}")
    print(f"Tiempo en segundos: {tiempo_total}")
    print(f"Tiempo en segundos GREEDY: {tiempo_greedy}")
    print(f"Tiempo en segundos BL: {tiempo_bl}")
    print(f"Tiempo en segundos Generacion Poblacion: {tiempo_poblacion}")
    print(f"Tiempo en segundos PMX: {tiempo_pmx}")

    print(f"NOTA: {nota_x}")


if __name__ == "__main__":
    main()

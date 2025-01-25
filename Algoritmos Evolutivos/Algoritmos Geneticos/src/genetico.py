import numpy as np
import random

def calcular_coste(matriz_flujo, matriz_distancia, individuo):
    matriz_flujo = np.asarray(matriz_flujo)
    matriz_distancia = np.asarray(matriz_distancia)
    individuo = np.asarray(individuo)

    permutacion = matriz_distancia[np.ix_(individuo, individuo)]

    return np.sum(matriz_flujo * permutacion)


def completar_hijo(hijo, padre, punto1, punto2):
    n = len(hijo)
    segmento = set(hijo[punto1:punto2 + 1])
    indices_vacios = [i for i in range(n) if hijo[i] == -1] 
    for i in indices_vacios:
        valor = padre[i]

        while valor in segmento:
            idx = padre.index(valor)  
            valor = padre[(idx + 1) % n]  
        hijo[i] = valor
        segmento.add(valor) 


def cruce_pmx(padre1, padre2):
    n = len(padre1)
    hijo1 = [-1] * n
    hijo2 = [-1] * n

    punto1, punto2 = sorted(random.sample(range(n), 2))

    hijo1[punto1:punto2 + 1] = padre2[punto1:punto2 + 1]
    hijo2[punto1:punto2 + 1] = padre1[punto1:punto2 + 1]

    completar_hijo(hijo1, padre1, punto1, punto2)
    completar_hijo(hijo2, padre2, punto1, punto2)

    return hijo1, hijo2


def mutacion(individuo, tamanio_matriz):
    i, j = random.sample(range(tamanio_matriz), 2)
    individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo

def seleccion_torneo(poblacion, matriz_flujo, matriz_distancia):
    ind1, ind2 = random.sample(range(len(poblacion)), 2)
    coste1 = calcular_coste(matriz_flujo, matriz_distancia, poblacion[ind1])
    coste2 = calcular_coste(matriz_flujo, matriz_distancia, poblacion[ind2])
    return poblacion[ind1] if coste1 < coste2 else poblacion[ind2]

def reemplazo_generacional(poblacion, hijos, matriz_flujo, matriz_distancia):
    poblacion_completa = poblacion + hijos
    poblacion_completa.sort(key=lambda ind: calcular_coste(matriz_flujo, matriz_distancia, ind))
    return poblacion_completa[:len(poblacion)]

def agg_pmx(poblacion, matriz_flujo, matriz_distancia, tamanio_matriz, iter_max, prob_cruce=0.7, prob_mutacion=0.01, semilla=None):
    if semilla is not None:
        random.seed(semilla)

    tamanio_poblacion = len(poblacion)

    #print(tamanio_matriz)

    for generacion in range(iter_max):

        print(f"Generacion: {generacion}")

        # Selección
        padres = [seleccion_torneo(poblacion, matriz_flujo, matriz_distancia) for _ in range(tamanio_poblacion)]

        #print(len(padres))

        # Cruce
        hijos = []
        for i in range(0, tamanio_poblacion, 2):
            if random.random() < prob_cruce:
                hijo1, hijo2 = cruce_pmx(padres[i], padres[i + 1])
                hijos.extend([hijo1, hijo2])
            else:
                hijos.extend([padres[i], padres[i + 1]])

        # Mutación
        for i in range(len(hijos)):
            if random.random() < prob_mutacion:
                hijos[i] = mutacion(hijos[i], tamanio_matriz)

        # Reemplazo
        poblacion = reemplazo_generacional(poblacion, hijos, matriz_flujo, matriz_distancia)

    mejor_individuo = min(poblacion, key=lambda ind: calcular_coste(matriz_flujo, matriz_distancia, ind))
    return mejor_individuo

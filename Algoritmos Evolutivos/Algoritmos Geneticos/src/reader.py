import os

class Reader:
    def __init__(self):
        self.matriz_flujo = None
        self.matriz_distancia = None
        self.tamanio_matrices = 0
        self.archivo = ""

    def establecer_nombre_archivo(self, archivo):
        self.archivo = archivo

    def leer_matrices(self):
        ruta_pre = "../datos/"
        ruta_def = os.path.join(ruta_pre, self.archivo)

        try:
            with open(ruta_def, "r") as archivo:
                self.tamanio_matrices = int(archivo.readline().strip())
                self.matriz_flujo = [[0] * self.tamanio_matrices for _ in range(self.tamanio_matrices)]
                self.matriz_distancia = [[0] * self.tamanio_matrices for _ in range(self.tamanio_matrices)]

                lineas = [linea.strip() for linea in archivo if linea.strip()]

                indice_linea = 0
                for i in range(self.tamanio_matrices):
                    valores = list(map(int, lineas[indice_linea].split()))
                    for j in range(self.tamanio_matrices):
                        self.matriz_flujo[i][j] = valores[j]
                    indice_linea += 1

                for i in range(self.tamanio_matrices):
                    valores = list(map(int, lineas[indice_linea].split()))
                    for j in range(self.tamanio_matrices):
                        self.matriz_distancia[i][j] = valores[j]
                    indice_linea += 1
        except FileNotFoundError:
            print(f"Archivo '{ruta_def}' no encontrado.")
        except Exception as e:
            print(f"Error al leer el archivo: {e}")


    def imprimir_matrices(self, matriz_flujo, matriz_distancia):
        if matriz_flujo is None or matriz_distancia is None:
            print("No se han cargado las matrices. Asegúrate de leer un archivo primero.")
            return

        print("Matriz Flujo:")
        for fila in matriz_flujo:
            print(" ".join(map(str, fila)))

        print("\nMatriz Distancia:")
        for fila in matriz_distancia:
            print(" ".join(map(str, fila)))

    def get_matriz_flujo(self):
        return self.matriz_flujo

    def get_matriz_distancia(self):
        return self.matriz_distancia

    def get_tamanio_matriz(self):
        return self.tamanio_matrices

    def get_nombre_archivo(self):
        return self.archivo

    def obtener_todos_archivos(self):
        ruta_carpeta = "../datos/"
        try:
            archivos = [f for f in os.listdir(ruta_carpeta) if os.path.isfile(os.path.join(ruta_carpeta, f))]
            return archivos
        except FileNotFoundError:
            print(f"Carpeta '{ruta_carpeta}' no encontrada.")
            return []
        except Exception as e:
            print(f"Error al listar archivos: {e}")
            return []

import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import deque
import numpy as np
import networkx as nx


class Cuadrado():
    def __init__(self, dimensiones=50, p=0.45, pared=4, cueva=5, semilla=None):
        self.semilla = semilla if semilla is not None else random.randint(0, 2 ** 32 - 1)
        random.seed(semilla)
        self.dimensiones = dimensiones
        self.pared = pared
        self.cueva = cueva
        self.mapa = [[True if random.random() < p else False for _ in range(dimensiones)] for _ in range(dimensiones)]

    def pintar_mapa(self):
        # Crear un colormap personalizado donde True será blanco y False será negro
        cmap = mcolors.ListedColormap(['blue', 'white'])

        plt.figure(figsize=(5, 5))
        plt.imshow(self.mapa, cmap=cmap, interpolation="nearest")
        plt.axis("off")
        plt.show()

    def paso_de_simulacion(self):
        temp_mapa = [[None for _ in range(self.dimensiones)] for _ in range(self.dimensiones)]
        for x in range(self.dimensiones):
            for y in range(self.dimensiones):
                vecinos_vivos = self.contar_vecinos_vivos(x, y)
                if self.mapa[x][y]:
                    if vecinos_vivos < self.pared:
                        temp_mapa[x][y] = False
                    else:
                        temp_mapa[x][y] = True
                else:
                    if vecinos_vivos > self.cueva:
                        temp_mapa[x][y] = True
                    else:
                        temp_mapa[x][y] = False

        self.mapa = temp_mapa

    def contar_vecinos_vivos(self, x, y):
        '''
            x-1 x x+1
        y-1  #  #  #
         y   #  x  #
        y+1  #  #  #
        '''
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                vecino_x = x + i
                vecino_y = y + j
                # No queremos contarnos a nosotros mmismos
                if i == 0 and j == 0:
                    continue
                # Para rellenar los bordes de la cueva, esto puede aplicarse.
                elif vecino_x < 0 or vecino_y < 0 or vecino_x >= self.dimensiones or vecino_y >= self.dimensiones:
                    count += 1
                elif self.mapa[vecino_x][vecino_y]:
                    count += 1
        return count

    def mantener_mayor_caverna(self):
        visitados = [[False for _ in range(self.dimensiones)] for _ in range(self.dimensiones)]
        mayor_caverna = []

        for i in range(self.dimensiones):
            for j in range(self.dimensiones):
                if not self.mapa[i][j] and not visitados[i][j]:
                    caverna_actual = []
                    cola = deque()
                    cola.append((i, j))
                    visitados[i][j] = True

                    while cola:
                        x, y = cola.popleft()
                        caverna_actual.append((x, y))

                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < self.dimensiones and 0 <= ny < self.dimensiones:
                                if not self.mapa[nx][ny] and not visitados[nx][ny]:
                                    visitados[nx][ny] = True
                                    cola.append((nx, ny))
                    # print(f"Actual {len(caverna_actual)} | Mayor {len(mayor_caverna)}")
                    if len(caverna_actual) > len(mayor_caverna):
                        mayor_caverna = caverna_actual
        nuevo_mapa = [[True for _ in range(self.dimensiones)] for _ in range(self.dimensiones)]
        for x, y in mayor_caverna:
            nuevo_mapa[x][y] = False
        self.mapa = nuevo_mapa

    def generar_salas_flood(self, num_semillas=15, visualizar=True):
        """
        Divide la caverna en salas mediante expansión simultánea desde semillas.
        """
        # Obtener celdas accesibles y dimensiones
        celdas_suelo = np.argwhere(np.logical_not(self.mapa))
        dim = self.dimensiones

        # Manejar caso sin celdas de suelo
        if len(celdas_suelo) == 0:
            return np.full((dim, dim), -1, dtype=int), []

        # Ajustar num_semillas si es necesario
        if len(celdas_suelo) < num_semillas:
            num_semillas = len(celdas_suelo)

        # Seleccionar semillas usando random para mantener coherencia con la semilla de la clase
        indices = random.sample(range(len(celdas_suelo)), num_semillas)
        semillas = celdas_suelo[indices]

        # Inicializar matriz de salas y cola BFS
        salas = np.full((dim, dim), -1, dtype=int)
        cola = deque()

        # Asignar semillas e iniciar expansión
        for idx, (x, y) in enumerate(semillas):
            salas[x, y] = idx
            cola.append((x, y, idx))

        # Direcciones de expansión (4-vecinos)
        direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Expansión simultánea
        while cola:
            x, y, region = cola.popleft()
            for dx, dy in direcciones:
                nx, ny = x + dx, y + dy
                if 0 <= nx < dim and 0 <= ny < dim:
                    if salas[nx, ny] == -1 and not self.mapa[nx][ny]:
                        salas[nx, ny] = region
                        cola.append((nx, ny, region))

        # Visualización
        if visualizar:
            plt.figure(figsize=(8, 8))
            plt.imshow(salas, cmap='tab20', interpolation='nearest')

            # Dibujar bordes entre regiones
            bordes = np.zeros((dim, dim))
            for x in range(dim):
                for y in range(dim):
                    if salas[x, y] != -1:
                        for dx, dy in direcciones:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < dim and 0 <= ny < dim:
                                if salas[nx, ny] != salas[x, y]:
                                    bordes[x, y] = 1

            plt.imshow(bordes, cmap='gray', alpha=0.3)
            plt.scatter(semillas[:, 1], semillas[:, 0], c='black', s=30, marker='X')
            plt.title(f'Expansión simultánea ({num_semillas} semillas)')
            plt.axis('off')
            plt.show()

    def convertir_a_grafo(self, salas):
        """
        Convierte la matriz de regiones `salas` en un grafo, con nodos en los centroides de cada región
        y aristas entre regiones contiguas.
        """
        dim = self.dimensiones
        grafo = nx.Graph()

        # Calcular centroides de cada región
        regiones = {}
        for x in range(dim):
            for y in range(dim):
                region = salas[x][y]
                if region != -1:
                    if region not in regiones:
                        regiones[region] = []
                    regiones[region].append((x, y))

        self.centros = {}
        for region, celdas in regiones.items():
            # Calcular centroide
            xs, ys = zip(*celdas)
            centroide = (sum(ys) / len(ys), sum(xs) / len(xs))  # Nota: (col, fila)
            grafo.add_node(region, pos=centroide)
            self.centros[region] = centroide

        # Añadir aristas entre regiones contiguas
        for x in range(dim):
            for y in range(dim):
                actual = salas[x][y]
                if actual == -1:
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < dim and 0 <= ny_ < dim:
                        vecino = salas[nx_][ny_]
                        if vecino != -1 and vecino != actual:
                            grafo.add_edge(actual, vecino)

        self.grafo = grafo

    def mostrar_mapa_y_grafo(self, salas):
        """
        Muestra el mapa de regiones con el grafo superpuesto (nodos en los centroides).
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))

        # Mostrar el mapa de regiones
        cmap = plt.cm.get_cmap('tab20')
        ax.imshow(salas, cmap=cmap, origin='upper')

        # Mostrar el grafo
        pos = nx.get_node_attributes(self.grafo, 'pos')
        nx.draw(self.grafo, pos, with_labels=True, node_color='red',
                edge_color='yellow', node_size=300, ax=ax, font_size=10)
        nx.draw_networkx_edge_labels(self.grafo, pos,
                                     edge_labels={(u, v): '' for u, v in self.grafo.edges},
                                     font_color='gray', ax=ax)

        ax.set_title("Mapa y grafo superpuesto")
        ax.axis('off')
        plt.show()

def automata_celular(iteraciones = 4,dimensiones = 50, p = 0.45, pared = 4, cueva = 5, num_semillas=5,semilla = None):
    mapa = Cuadrado(dimensiones=dimensiones, p = p, pared = pared, cueva = cueva, semilla = semilla)
    for _ in range(iteraciones):
        mapa.paso_de_simulacion()
    mapa.mantener_mayor_caverna()
    mapa.generar_salas_flood(num_semillas=num_semillas)
    #mapa.pintar_mapa()
    return mapa

iteraciones = 4
dimensiones = 50
p = 0.6
pared = 5
cueva = 5
semilla = None
mapa = automata_celular(iteraciones=iteraciones, dimensiones=dimensiones, p=p, pared=pared, cueva=cueva, num_semillas = 5, semilla=semilla)
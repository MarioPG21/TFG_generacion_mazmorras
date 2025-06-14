import networkx as nx
import statistics
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from matplotlib.colors import ListedColormap
from collections import deque
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import time
import pandas as pd
import traceback
from itertools import product
"""
------------------------------------------------------------------------------------------------------------------------
                                                    RANDOM PLACEMENT
------------------------------------------------------------------------------------------------------------------------
"""
def encontrar_mas_cercana(salas, nuevo_centro_x, nuevo_centro_y):
    cercana = None
    min_dist = float('inf')
    for sala in salas:
        centro_x, centro_y = sala.get_centro()
        distancia = distancia_cuadrado(centro_x, centro_y, nuevo_centro_x, nuevo_centro_y)
        if distancia < min_dist:
            min_dist = distancia
            cercana = sala
    return cercana

def distancia_cuadrado(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

class Sala:
    def __init__(self, x, y, lado):
        self.x1 = x
        self.y1 = y
        self.x2 = x + lado
        self.y2 = y + lado

    def get_centro(self):
        centro_x = (self.x1 + self.x2) // 2
        centro_y = (self.y1 + self.y2) // 2
        return (centro_x, centro_y)

    def se_superpone(self, otra):
        return (self.x1 <= otra.x2 and self.x2 >= otra.x1 and
                self.y1 <= otra.y2 and self.y2 >= otra.y1)

class Dungeon:
    def __init__(self, anchura, altura, max_salas, semilla=None, intentos_por_sala=10):
        self.semilla = semilla if semilla is not None else random.randint(0, 2 ** 32 - 1)
        random.seed(self.semilla)
        self.mazmorra = [[0 for _ in range(anchura)] for _ in range(altura)]
        self.grafo = nx.Graph()
        self.centros = {}
        salas = []
        self.puertas = []

        for _ in range(max_salas):
            for _ in range(intentos_por_sala):
                lado = 5
                x = random.randint(0, anchura - lado - 1)
                y = random.randint(0, altura - lado - 1)
                nueva_sala = Sala(x, y, lado)

                if any(nueva_sala.se_superpone(otra_sala) for otra_sala in salas):
                    continue  # La sala se superpone, se descarta este intento

                sala_id = len(salas)
                self.centros[sala_id] = nueva_sala.get_centro()
                self.grafo.add_node(sala_id)

                if salas:
                    nuevo_centro_x, nuevo_centro_y = nueva_sala.get_centro()
                    cercana = encontrar_mas_cercana(salas, nuevo_centro_x, nuevo_centro_y)
                    cercana_id = salas.index(cercana)
                    cercana_centro_x, cercana_centro_y = cercana.get_centro()

                    self.crear_pasillo_horizontal(cercana_centro_x, nuevo_centro_x, cercana_centro_y)
                    self.crear_pasillo_vertical(cercana_centro_y, nuevo_centro_y, nuevo_centro_x)

                salas.append(nueva_sala)
                for x in range(nueva_sala.x1, nueva_sala.x2):
                    for y in range(nueva_sala.y1, nueva_sala.y2):
                        self.mazmorra[y][x] = 1  # Sala

                break  # Si se colocó correctamente, no hace falta seguir intentando

        self._construir_grafo_desde_matriz(salas)

    def crear_pasillo_horizontal(self, x1, x2, y):
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if self.mazmorra[y][x] == 0:
                self.mazmorra[y][x] = 2  # Pasillo

    def crear_pasillo_vertical(self, y1, y2, x):
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if self.mazmorra[y][x] == 0:
                self.mazmorra[y][x] = 2  # Pasillo

    def _construir_grafo_desde_matriz(self, salas):
        altura = len(self.mazmorra)
        anchura = len(self.mazmorra[0])
        mapa_salas = [[-1 for _ in range(anchura)] for _ in range(altura)]

        # 1. Mapear celdas de cada sala
        for i, sala in enumerate(salas):
            for x in range(sala.x1, sala.x2):
                for y in range(sala.y1, sala.y2):
                    mapa_salas[y][x] = i

        conectados = set()

        # 2. Para cada sala, buscar pasillos contiguos y hacer BFS solo por pasillos
        for i, sala in enumerate(salas):
            visitado = [[False for _ in range(anchura)] for _ in range(altura)]
            cola = deque()

            # Añadir las celdas frontera de la sala que estén pegadas a pasillo
            for x in range(sala.x1, sala.x2):
                for y in range(sala.y1, sala.y2):
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < anchura and 0 <= ny < altura and self.mazmorra[ny][nx] == 2:
                            cola.append((nx, ny))
                            visitado[ny][nx] = True

            while cola:
                x, y = cola.popleft()

                # Revisar vecinos
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < anchura and 0 <= ny < altura):
                        continue
                    if visitado[ny][nx]:
                        continue

                    if self.mazmorra[ny][nx] == 2:
                        cola.append((nx, ny))
                        visitado[ny][nx] = True

                    elif self.mazmorra[ny][nx] == 1:
                        j = mapa_salas[ny][nx]
                        if j != -1 and j != i:
                            par = tuple(sorted((i, j)))
                            if par not in conectados:
                                distancia = ((self.centros[i][0] - self.centros[j][0]) ** 2 +
                                             (self.centros[i][1] - self.centros[j][1]) ** 2) ** 0.5
                                self.grafo.add_edge(i, j, weight=distancia)
                                conectados.add(par)

    def mostrar(self, guardar=False):
        cmap = ListedColormap(['black', 'white', 'gray'])
        plt.figure(figsize=(10, 6))
        plt.imshow(self.mazmorra, cmap=cmap, origin='upper')

        # Cambiar título a la semilla
        titulo = f"{self.semilla}"
        plt.title(titulo)

        plt.axis('off')

        # Guardar si se indica
        if guardar:
            carpeta = "Mazmorras_generadas/Colocacion_aleatoria"
            os.makedirs(carpeta, exist_ok=True)
            path = os.path.join(carpeta, f"ColAl_{self.semilla}.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
            #print(f"Imagen guardada en: {path}")

        plt.show()

    def mostrar_grafo(self):
        pos = self.centros
        labels = nx.get_edge_attributes(self.grafo, 'weight')
        nx.draw(self.grafo, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
        nx.draw_networkx_edge_labels(self.grafo, pos, edge_labels={k: f"{v:.1f}" for k, v in labels.items()})
        plt.title("Grafo de conexión entre salas")
        plt.show()

    def mostrar_mapa_y_grafo(self, guardar=False):
        fig, ax = plt.subplots(figsize=(10, 6))

        # 1. Mostrar matriz con colormap
        cmap = ListedColormap(['black', 'white', 'gray'])  # 0 = pared, 1 = sala, 2 = pasillo
        ax.imshow(self.mazmorra, cmap=cmap, origin='upper')

        # 2. Dibujar grafo con nodos en el centro de cada sala
        pos = {k: (v[0], v[1]) for k, v in self.centros.items()}
        labels = nx.get_edge_attributes(self.grafo, 'weight')

        nx.draw(self.grafo, pos, with_labels=True, node_color='red', edge_color='yellow', node_size=400, ax=ax)
        nx.draw_networkx_edge_labels(self.grafo, pos, edge_labels={k: f"{v:.0f}" for k, v in labels.items()}, ax=ax)

        ax.set_title("Mazmorra y grafo superpuesto")
        ax.axis('off')

        # Guardar si se indica
        if guardar:
            carpeta = "Mazmorras_generadas/Colocacion_aleatoria"
            os.makedirs(carpeta, exist_ok=True)
            path = os.path.join(carpeta, f"ColAl_{self.semilla}_grafo.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
            #print(f"Imagen guardada en: {path}")
        plt.show()

def generar_dungeon_ColAl(anchura=60, altura=30, max_salas=10, semilla=None, intentos_por_sala=10, guardar=False):
    dungeon = Dungeon(anchura, altura, max_salas, semilla=semilla, intentos_por_sala=intentos_por_sala)
    dungeon.mostrar(guardar = guardar)
    dungeon.mostrar_mapa_y_grafo(guardar = guardar)
    m=dungeon.mazmorra
    g=dungeon.grafo
    return m,g

"""
------------------------------------------------------------------------------------------------------------------------
                                                    CELLULAR AUTOMATA
------------------------------------------------------------------------------------------------------------------------
"""
class Mapa():
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

    def generar_salas_flood(self, num_semillas=15, guardar=False):
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

        fig, ax = plt.subplots(figsize=(8, 8))

        # Base: blanco para cueva, negro para pared
        base = np.where(self.mapa, 0, 1)
        ax.imshow(base, cmap=ListedColormap(['black', 'white']), origin='upper')

        # Dibujar bordes entre salas distintas (pero no con pared)
        bordes = np.zeros((dim, dim))
        for x in range(dim):
            for y in range(dim):
                if not self.mapa[x][y] and salas[x, y] != -1:
                    for dx, dy in direcciones:
                        nx_, ny_ = x + dx, y + dy
                        if 0 <= nx_ < dim and 0 <= ny_ < dim:
                            if not self.mapa[nx_][ny_] and salas[nx_, ny_] != -1:
                                if salas[nx_, ny_] != salas[x, y]:
                                    bordes[x, y] = 1

        ax.imshow(np.where(bordes == 1, 0.5, np.nan), cmap=ListedColormap(['gray']), alpha=1.0, origin='upper')

        # Marcar semillas opcional
        ax.scatter(semillas[:, 1], semillas[:, 0], c='red', s=20, marker='x')

        ax.set_title(str(self.semilla))
        ax.axis('off')

        if guardar:
            carpeta = "Mazmorras_generadas/Automata_Celular"
            os.makedirs(carpeta, exist_ok=True)
            path = os.path.join(carpeta, f"AuCel_{self.semilla}.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
            #print(f"Imagen guardada en: {path}")

        plt.show()

        return salas

    def matriz_a_grafo(self, salas):
        """
        Crea un grafo desde la matriz de regiones (salas) generadas por flood fill.
        Cada nodo corresponde al centroide de una sala.
        Las aristas se crean entre regiones adyacentes.
        """
        dim = self.dimensiones
        grafo = nx.Graph()
        regiones = {}

        # Agrupar celdas por región
        for x in range(dim):
            for y in range(dim):
                r = salas[x][y]
                if r != -1:
                    regiones.setdefault(r, []).append((x, y))

        # Añadir nodos en los centroides
        self.centros = {}
        for region, celdas in regiones.items():
            xs, ys = zip(*celdas)
            centroide = (sum(ys) / len(ys), sum(xs) / len(xs))  # (columna, fila)
            grafo.add_node(region, pos=centroide)
            self.centros[region] = centroide

        # Añadir aristas entre regiones adyacentes
        ya_conectados = set()
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
                            par = tuple(sorted((actual, vecino)))
                            if par not in ya_conectados:
                                ya_conectados.add(par)
                                grafo.add_edge(*par)
        self.grafo = grafo

    def mostrar_mapa_y_grafo(self, salas, guardar=False):
        """
        Muestra el mapa binario (cueva/pared) y bordes solo entre salas diferentes, con grafo superpuesto.
        """
        dim = self.dimensiones
        fig, ax = plt.subplots(figsize=(8, 8))

        # Base: blanco para cueva, negro para pared
        base = np.where(self.mapa, 0, 1)
        ax.imshow(base, cmap=ListedColormap(['black', 'white']), origin='upper')

        # Bordes solo entre salas distintas
        bordes = np.zeros((dim, dim))
        for x in range(dim):
            for y in range(dim):
                if not self.mapa[x][y] and salas[x, y] != -1:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx_, ny_ = x + dx, y + dy
                        if 0 <= nx_ < dim and 0 <= ny_ < dim:
                            if not self.mapa[nx_][ny_] and salas[nx_, ny_] != -1:
                                if salas[nx_, ny_] != salas[x, y]:
                                    bordes[x, y] = 1

        ax.imshow(np.where(bordes == 1, 0.5, np.nan), cmap=ListedColormap(['gray']), alpha=1.0, origin='upper')

        # Mostrar grafo
        pos = nx.get_node_attributes(self.grafo, 'pos')
        nx.draw(self.grafo, pos, with_labels=True, node_color='red',
                edge_color='yellow', node_size=300, ax=ax, font_size=10)
        nx.draw_networkx_edge_labels(self.grafo, pos,
                                     edge_labels={(u, v): '' for u, v in self.grafo.edges},
                                     font_color='gray', ax=ax)

        ax.set_title("Mapa y grafo (solo bordes entre salas)")
        ax.axis('off')

        if guardar:
            carpeta = "Mazmorras_generadas/Automata_Celular"
            os.makedirs(carpeta, exist_ok=True)
            path = os.path.join(carpeta, f"AuCel_{self.semilla}_grafo.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
            #print(f"Imagen guardada en: {path}")

        plt.show()

def automata_celular(iteraciones=4, dimensiones=50, p=0.45, pared=4, cueva=5, num_semillas=5, semilla=None, guardar=False):
    mapa = Mapa(dimensiones=dimensiones, p=p, pared=pared, cueva=cueva, semilla=semilla)
    for _ in range(iteraciones):
        mapa.paso_de_simulacion()
    #mapa.pintar_mapa()
    mapa.mantener_mayor_caverna()
    salas = mapa.generar_salas_flood(num_semillas=10, guardar=guardar)

    mapa.matriz_a_grafo(salas)
    mapa.mostrar_mapa_y_grafo(salas=salas, guardar=guardar)
    m=mapa.mapa
    g=mapa.grafo
    return m,g
"""
------------------------------------------------------------------------------------------------------------------------
                                                    SPACE PARTITIONING
------------------------------------------------------------------------------------------------------------------------
"""
def formar_particion_estado_4(n, d=5, semilla=None):
    # Para conseguir reproducibilidad, podemos introducir una semilla al algoritmo.
    if semilla is not None:
        random.seed(semilla)
    else:
        # Generar una seed aleatoria y almacenarla
        semilla = random.randint(0, 2 ** 32 - 1)
        random.seed(semilla)  # Establecer la seed

    lista_hojas = {1}  # Usamos un set para mejor eficiencia
    A = nx.DiGraph()
    A.add_node(1, profundidad=0)
    A.graph['altura'] = 0
    nodos = 1

    for _ in range(n):
        if lista_hojas:
            selec = random.choice(tuple(lista_hojas))  # Convertir a tupla para elección aleatoria rápida
            profundidad_hijos = A.nodes[selec]["profundidad"] + 1

            if profundidad_hijos > A.graph['altura']:
                A.graph['altura'] = profundidad_hijos

            # Agregar los nuevos nodos
            for j in range(1, 5):
                nuevo_nodo = nodos + j
                A.add_node(nuevo_nodo, profundidad=profundidad_hijos)
                A.add_edge(selec, nuevo_nodo)
                if profundidad_hijos < d:
                    lista_hojas.add(nuevo_nodo)

            lista_hojas.remove(selec)  # El nodo ya no es hoja
            nodos += 4  # Se añaden 4 nodos en cada iteración
        else:
            break

    return A, semilla
class Cuadrado():
    def __init__(self, G, nodo=1, centro=None, tamaño=10, semilla=None):

        self.semilla = semilla if semilla is not None else random.randint(0, 2 ** 32 - 1)
        random.seed(self.semilla)

        self.G = G

        self.nodo = nodo
        self.tamaño = tamaño
        if centro is None:
            self.centro = (self.tamaño / 2, self.tamaño / 2)
        else:
            self.centro = centro
        self.profundidad = self.G.graph['altura']

        self.tipo = 'pared'
        self.nsalas = 0
        self.salas = []

        self.caminos = []

        # limites = [arriba,derecha,abajo,izquierda]
        offset = self.tamaño / 2
        self.limites = [
            self.centro[1] + offset,
            self.centro[0] + offset,
            self.centro[1] - offset,
            self.centro[0] - offset
        ]

        self.cuadrantes = []

        if self.nodo in self.G:
            hijos = list(self.G.successors(nodo))
        else:
            hijos = []
        if hijos:
            # Se divide por cuatro porque tiene la mitad de tamaño, y después se divide a la mitad otra vez para determinar el centro.
            mitad_tamaño = self.tamaño / 2
            centro_cuadrante = mitad_tamaño / 2
            offsets = [
                (-centro_cuadrante, centro_cuadrante),  # Cuadrante arriba izquierda
                (centro_cuadrante, centro_cuadrante),  # Cuadrante arriba derecha
                (-centro_cuadrante, -centro_cuadrante),  # Cuadrante abajo izquierda
                (centro_cuadrante, -centro_cuadrante)  # Cuadrante abajo derecha
            ]

            for hijo, offset in zip(hijos, offsets):
                nuevo_centro = (self.centro[0] + offset[0], self.centro[1] + offset[1])
                self.cuadrantes.append(Cuadrado(self.G, hijo, nuevo_centro, mitad_tamaño, semilla=self.semilla))


    def __str__(self):
        return f"Cuadrado(nodo={self.nodo},tipo={self.tipo})"

    def formar_grafo(self):
        lista_hojas = self.hojas_preorden()
        self.S = nx.Graph()
        # Preprocesar cuadrantes para evitar múltiples llamadas a encontrar_cuadrante()
        cuadrantes = {i: self.encontrar_cuadrante(i) for i in lista_hojas}

        # Preprocesar nodos en el grafo S
        for i, cuadrante in cuadrantes.items():
            if cuadrante.nodo not in self.S:
                self.S.add_node(cuadrante.nodo,
                                arriba=[], derecha=[], izquierda=[], abajo=[],
                                tamaño=self.profundidad - self.G.nodes[cuadrante.nodo]['profundidad'])

        # Direcciones y sus índices en self.limites
        direcciones = {
            'arriba': (0, 2, 'abajo'),
            'derecha': (1, 3, 'izquierda'),
            'abajo': (2, 0, 'arriba'),
            'izquierda': (3, 1, 'derecha')
        }

        # Comparar cada par de nodos
        for i, yo in cuadrantes.items():
            for j, tu in cuadrantes.items():
                if i == j:
                    continue  # Evitar comparar el mismo nodo consigo mismo

                for direccion, (lim_i, lim_j, opuesta) in direcciones.items():
                    if yo.limites[lim_i] == tu.limites[lim_j]:
                        if (direccion in ['arriba', 'abajo'] and yo.limites[1] >= tu.limites[1] and yo.limites[3] <=
                            tu.limites[3]) or \
                                (direccion in ['derecha', 'izquierda'] and yo.limites[0] <= tu.limites[0] and
                                 yo.limites[2] >= tu.limites[2]):

                            self.S.add_edge(tu.nodo, yo.nodo)

                            if tu.nodo not in self.S.nodes[yo.nodo][direccion]:
                                self.S.nodes[yo.nodo][direccion].append(tu.nodo)
                            if yo.nodo not in self.S.nodes[tu.nodo][opuesta]:
                                self.S.nodes[tu.nodo][opuesta].append(yo.nodo)

    def encontrar_cuadrante(self, id):
        """Busca el cuadrante que contiene el nodo con el ID especificado."""
        if self.nodo == id:
            # print(f"Nodo encontrado: {self.nodo}")
            return self

        for cuadrante in self.cuadrantes:
            resultado = cuadrante.encontrar_cuadrante(id)
            if resultado is not None:
                return resultado

        return None

    def hojas_preorden(self):
        return [
            nodo for nodo in nx.dfs_preorder_nodes(self.G, source=1)
            if self.G.out_degree(nodo) == 0
        ]

    def crear_mazmorra_clasica(self, n=5, densidad=1):
        def generar_expansiones(nodos, selec):
            direcciones_pequeño = set()
            direcciones_mismo = set()
            pares_contiguos = {("arriba", "derecha"), ("derecha", "abajo"), ("abajo", "izquierda"),
                               ("izquierda", "arriba"), }
            # E00
            expansiones = {"E00": [selec]}

            for attr, vecinos in nodos[selec].items():
                if attr == 'tamaño':
                    continue  # Saltar la clave 'tamaño'

                tamaño_menor = []
                tamaño_igual = []

                for nodo in vecinos:
                    if nodos[nodo]['tamaño'] == nodos[selec]['tamaño'] - 1:
                        tamaño_menor.append(nodo)
                    elif nodos[nodo]['tamaño'] == nodos[selec]['tamaño']:
                        tamaño_igual.append(nodo)

                if len(tamaño_menor) == 2:
                    direcciones_pequeño.add(attr)
                    expansiones[f"E01_{attr}"] = [selec] + tamaño_menor

                if tamaño_igual:
                    direcciones_mismo.add(attr)
                    expansiones[f"E04_{attr}"] = [selec] + tamaño_igual

            arriba = nodos[selec]['arriba']
            abajo = nodos[selec]['abajo']
            mapa_indices = {
                ("arriba", "derecha"): (arriba, 'derecha', 1),
                ("derecha", "abajo"): (abajo, 'derecha', 1),
                ("abajo", "izquierda"): (abajo, 'izquierda', 0),
                ("izquierda", "arriba"): (arriba, 'izquierda', 0)
            }
            # Direcciones de expansión contiguas
            for d1, d2 in pares_contiguos:

                if {d1, d2}.issubset(direcciones_pequeño):
                    expansiones[f"E02_{d1}_{d2}"] = [selec] + nodos[selec][d1] + nodos[selec][d2]
                    if (d1, d2) in mapa_indices:
                        nodo_y, nodo_x, indice = mapa_indices[(d1, d2)]
                        esquina = nodos[nodo_y[indice]][nodo_x][0]
                        if nodos[esquina]['tamaño'] == nodos[selec]['tamaño'] - 1:
                            expansiones[f"E03_{d1}_{d2}"] = [selec] + nodos[selec][d1] + nodos[selec][d2] + [esquina]

                if {d1, d2}.issubset(direcciones_mismo):
                    expansiones[f"E05_{d1}_{d2}"] = [selec] + nodos[selec][d1] + nodos[selec][d2]
                    if (d1, d2) in mapa_indices:
                        nodo_y, nodo_x, indice = mapa_indices[(d1, d2)]
                        esquina = nodos[nodo_y[0]][nodo_x][0]
                        if nodos[esquina]['tamaño'] == nodos[selec]['tamaño']:
                            expansiones[f"E03_{d1}_{d2}"] = [selec] + nodos[selec][d1] + nodos[selec][d2] + [esquina]

            if {"arriba", "derecha", "abajo", "izquierda"}.issubset(direcciones_mismo):
                esquinas = {
                    "arriba_derecha": nodos[arriba[0]]['derecha'][0],
                    "derecha_abajo": nodos[abajo[0]]['derecha'][0],
                    "abajo_izquierda": nodos[abajo[0]]['izquierda'][0],
                    "izquierda_arriba": nodos[arriba[0]]['izquierda'][0]
                }
                del esquina
                if all(nodos[esquina]['tamaño'] == nodos[selec]['tamaño'] for esquina in esquinas.values()):
                    expansiones[f"E07"] = (
                            [selec] +
                            nodos[selec]['arriba'] + nodos[selec]['derecha'] + nodos[selec]['abajo'] + nodos[selec][
                                'izquierda'] +
                            list(esquinas.values()))

            return dict(sorted((expansiones.items())))

        def seleccionar_expansion(expansiones):
            # TODO: selección por tamaño. Relacionado con densidad?
            if not expansiones:
                return False
            while expansiones:
                expansion = random.choice(list(expansiones.keys()))
                # print(f"\tExpansion: {expansion}")
                if all(self.comprobar_adyacentes(nodo, densidad) for nodo in expansiones[expansion]):
                    # print(f"\t\tEs válida")
                    for nodo in expansiones[expansion]:
                        if nodo in lista_hojas:
                            lista_hojas.remove(nodo)
                        self.encontrar_cuadrante(nodo).tipo = 'suelo'
                    self.nsalas += 1
                    self.salas.append(expansiones[expansion])
                    # print(f"SELECCIONADO {selec} con la expansión {expansion}\n")
                    return True
                # print(f"\t\tNo es válida")
                del expansiones[expansion]
                # print(f"\tExpansión: {expansion}")
            return False

        def grafo_desde_arbol(cuadrado, graph, tamaño):
            """Rellena el grafo con valores desde un quadtree, invirtiendo el eje Y."""

            if cuadrado is None:
                return

            lado = 2 ** (cuadrado.profundidad + 1)
            step = lado / tamaño
            lado_cuadrado = tamaño / lado
            if not cuadrado.cuadrantes:  # Es una hoja
                value = 0 if cuadrado.tipo == 'pared' else 1

                # Convertir coordenadas del cuadrado a índices del grafo
                x_start = int(cuadrado.limites[3] * step)
                x_end = int(cuadrado.limites[1] * step)
                y_start = int(cuadrado.limites[2] * step)  # Índices invertidos para Y
                y_end = int(cuadrado.limites[0] * step)

                # Crear nodos en el grafo para la celda
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        # Calcular centro y tamaño como en Cuadrado
                        centro_x = x * lado_cuadrado + lado_cuadrado / 2  # Centro en X
                        centro_y = (y * lado_cuadrado + lado_cuadrado / 2)  # Inversión Y para el sistema del Cuadrado
                        dimensiones = {
                            'centro': (centro_x, centro_y),
                            'tamaño': tamaño / lado
                        }
                        # Crear el nodo con coordenadas (x, y) como identificador
                        graph.add_node((x, y), value=value, nodo=cuadrado.nodo, dimensiones=dimensiones)

                # Conectar los nodos adyacentes si es necesario
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        # Conectar con el nodo a la derecha
                        if x < lado - 1:
                            graph.add_edge((x, y), (x + 1, y), weight=1)
                        # Conectar con el nodo abajo (invertido)
                        if y > 0:
                            graph.add_edge((x, y), (x, y - 1), weight=1)  # Conectar con el nodo de abajo
                        # Conectar con el nodo de arriba (invertido)
                        if y < lado - 1:
                            graph.add_edge((x, y), (x, y + 1), weight=1)  # Conectar con el nodo de arriba

            else:  # Llamar recursivamente a los hijos
                for child in cuadrado.cuadrantes:
                    grafo_desde_arbol(child, graph, tamaño)

        def ajustar_pesos(graph):
            # Convertir nodos a conjunto para acceso rápido
            nodos = {node for node in graph.nodes if graph.nodes[node]['value'] == 1}
            # Obtener bordes (vecinos de nodos que no están en nodos)
            bordes = {vecino for nodo in nodos for vecino in graph.neighbors(nodo) if vecino not in nodos}

            # Ajustar pesos de las aristas
            for u, v in graph.edges():
                if u in nodos and v in nodos:
                    graph.edges[u, v]['weight'] = 10
                elif u in nodos or v in nodos or u in bordes or v in bordes:
                    graph.edges[u, v]['weight'] = 5

        def conectar_salas(cuadrado):
            def encontrar_camino_entre_salas(graph, sala_a, sala_b):
                # Encontrar nodos más cercanos entre dos salas
                nodos_fuente = [n for n, attr in graph.nodes(data=True) if
                                attr['nodo'] in sala_a and attr['value'] == 1]
                nodos_destino = [n for n, attr in graph.nodes(data=True) if
                                 attr['nodo'] in sala_b and attr['value'] == 1]

                mejor_par = None
                min_distancia = float('inf')

                for fuente_nodo in nodos_fuente:
                    for destino_nodo in nodos_destino:
                        # Calcular distancia euclidiana entre coordenadas (x,y)
                        distancia = ((fuente_nodo[0] - destino_nodo[0]) ** 2 +
                                     (fuente_nodo[1] - destino_nodo[1]) ** 2)

                        if distancia < min_distancia:
                            min_distancia = distancia
                            mejor_par = (fuente_nodo, destino_nodo)

                return nx.shortest_path(graph, mejor_par[0], mejor_par[1], weight='weight')

            def marcar_camino(graph, camino):
                for nodo in camino[1:-1]:
                    graph.nodes[nodo]['value'] = 2
                    self.caminos.append(nodo)
                    for vecino in graph.neighbors(nodo):
                        graph.edges[nodo, vecino]['weight'] = 5

            graph = cuadrado.C
            salas = cuadrado.salas

            G_salas = nx.Graph()

            for i in range(len(salas)):
                for j in range(i + 1, len(salas)):
                    camino = encontrar_camino_entre_salas(graph, salas[i], salas[j])
                    peso_total = sum(graph.edges[(camino[k], camino[k + 1])]['weight'] for k in range(len(camino) - 1))
                    G_salas.add_edge(i, j, weight=peso_total)

            mst = nx.minimum_spanning_tree(G_salas)
            self.grafo_salas = mst
            for edge in mst.edges():
                fuente = salas[edge[0]]
                destino = salas[edge[1]]
                camino = encontrar_camino_entre_salas(graph, fuente, destino)
                marcar_camino(graph, camino)

        lista_hojas = set(self.hojas_preorden())
        nodos = self.S.nodes
        max_intentos = 100

        for i in range(n):
            intentos = 0
            while intentos < max_intentos:
                if not lista_hojas:
                    # print("No hay más hojas disponibles")
                    return

                selec = random.choice(list(lista_hojas))
                nodo_selec = self.encontrar_cuadrante(selec)
                # print(f"Nodo {selec} seleccionado")

                if nodo_selec.tipo != 'suelo':
                    # print("\tNo era suelo")
                    expansiones = generar_expansiones(nodos, selec)
                    # print(f"\tExpansiones: {expansiones}")

                    if seleccionar_expansion(expansiones):
                        break

                intentos += 1
            if intentos >= max_intentos:
                #print("Límite alcanzado")
                #print(f"Salas creadas: {i}")
                break

        self.C = nx.Graph()
        grafo_desde_arbol(self, self.C, self.tamaño)
        ajustar_pesos(self.C)
        conectar_salas(self)
        # mostrar_grafo(self.C)

    def comprobar_adyacentes(self, nodo, densidad=1):
        # Si la profundidad llega a 0, no se revisan más vecinos
        if densidad == 0:
            return True
        # print(f"Evaluando nodo {nodo}")
        # Se obtiene la lista de vecinos del nodo
        for vecino in list(self.S.neighbors(nodo)):
            # print(f"\tSu vecino {vecino}...")
            cuadrante = self.encontrar_cuadrante(vecino)

            # Si el vecino directo es 'suelo', retornamos False
            if cuadrante.tipo == 'suelo':
                # print("...es culpable.")
                return False
            # print("...está libre.")
            # Llamada recursiva para los vecinos de los vecinos, con la profundidad decrementada
            if not self.comprobar_adyacentes(vecino, densidad - 1):
                return False

        # Si todos los vecinos y vecinos de los vecinos hasta la profundidad son válidos, retornamos True
        return True

    def crear_mazmorra_dijkstra(self, n=5):
        pintados = []
        lista_hojas = self.hojas_preorden()
        print(f"Lista de hojas: {lista_hojas}")

        inicio = random.choice(lista_hojas)
        fin = random.choice(lista_hojas)
        while inicio == fin:
            fin = random.choice(lista_hojas)

        print(f"Camino de {inicio} a {fin}:")
        camino = nx.shortest_path(self.S, source=inicio, target=fin)
        print(camino)

        for nodo in camino:
            pintados.append(nodo)
            print(f"\tNodo: {nodo} pintado")
            cuadrante = self.encontrar_cuadrante(nodo)
            cuadrante.tipo = 'suelo'

        for i in range(n - 1):
            inicio = random.choice(lista_hojas)
            fin = random.choice(pintados)
            while inicio == fin:
                fin = random.choice(pintados)

            print(f"Camino de {inicio} a {fin}:")
            camino = nx.shortest_path(self.S, source=inicio, target=fin)
            print(camino)

            for nodo in camino:
                print(f"\tNodo: {nodo} pintado")
                cuadrante = self.encontrar_cuadrante(nodo)
                cuadrante.tipo = 'suelo'

    def obtener_matriz(self):
        max_x = max(x for (x, y) in self.C.nodes)
        max_y = max(y for (x, y) in self.C.nodes)
        matriz = [[0 for _ in range(max_x + 1)] for _ in range(max_y + 1)]

        for (x, y), data in self.C.nodes(data=True):
            matriz[y][x] = data['value']  # 0 = pared, 1 = sala, 2 = camino

        return matriz

    def imprimir_estructura(self, nivel=0):
        """
        Imprime la estructura jerárquica del cuadrado.
        """
        print(" " * nivel * 4 + str(self))
        for cuadrante in self.cuadrantes:
            cuadrante.imprimir_estructura(nivel + 1)

    def dibujar_grafo(self):
        """ Dibuja el grafo de conectividad de las salas con los pesos de las aristas """
        pos = {nodo: (self.encontrar_cuadrante(nodo).centro) for nodo in self.S.nodes}

        plt.figure(figsize=(self.tamaño, self.tamaño))

        # Dibujar nodos y aristas
        nx.draw(self.S, pos, with_labels=True, node_size=500, node_color="lightblue", edge_color="grey")

        # Obtener etiquetas de los pesos de las aristas
        edge_labels = {(u, v): self.S[u][v].get('peso', '') for u, v in self.S.edges}

        # Dibujar etiquetas de los pesos
        nx.draw_networkx_edge_labels(self.S, pos, edge_labels=edge_labels, font_color='red')

        plt.show()

    def dibujar_debug(self, guardar=False):
        # Colores constantes
        colores = {
            'pared': 'black',
            'suelo': 'white',
            'borde': 'None'  # Por defecto, sin borde
        }

        # Función auxiliar para dibujar los cuadrantes
        def dibujar_aux(raiz, ax, transparente):
            if not raiz.cuadrantes:
                # Dibuja el cuadrado actual
                x, y = raiz.centro
                tamaño = raiz.tamaño
                colores['borde'] = 'grey' if transparente else 'None'  # Cambia solo si es transparente

                rect = patches.Rectangle(
                    (x - tamaño / 2, y - tamaño / 2), tamaño, tamaño,
                    linewidth=1, edgecolor=colores['borde'], facecolor=colores[raiz.tipo]
                )
                ax.add_patch(rect)

                if transparente:
                    ax.text(x, y, str(raiz.nodo), color='grey', fontsize=raiz.tamaño * 5, ha='center', va='center')

            # Dibujar los hijos recursivamente
            for cuadrante in raiz.cuadrantes:
                dibujar_aux(cuadrante, ax, transparente)

        # Crear los subgráficos
        fig, (ax1, ax2) = plt.subplots(figsize=(self.tamaño * 2, self.tamaño), ncols=2)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(0.2)  # Fondo de la figura transparente

        # Dibujar ambos subgráficos con el parámetro adecuado de transparencia
        for ax, transparente in [(ax1, True), (ax2, False)]:
            ax.set_xlim(0, self.tamaño)
            ax.set_ylim(0, self.tamaño)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(str(self.semilla))
            dibujar_aux(self, ax, transparente)

        # Guardar la figura si es necesario
        if guardar:
            carpeta = "Cuadrados"
            os.makedirs(carpeta, exist_ok=True)  # Crear la carpeta solo si no existe

            # Guardar la figura
            archivo_guardado = os.path.join(
                carpeta, f"figura_{self.semilla}_SpacePart_{self.nsalas}.svg"
            )
            fig.savefig(archivo_guardado, transparent=False, bbox_inches='tight', dpi=300)

            #print(f"Figura guardada en: {archivo_guardado}")

    def dibujar(self, guardar=False):
        # Colores constantes
        colores = {
            'pared': 'black',
            'suelo': 'white',
            'camino': 'grey',
            'borde': 'None'  # Sin borde
        }

        # Función auxiliar para dibujar los cuadrantes
        def dibujar_aux(raiz, ax):
            if not raiz.cuadrantes:
                # Dibuja el cuadrado actual
                x, y = raiz.centro
                tamaño = raiz.tamaño

                rect = patches.Rectangle(
                    (x - tamaño / 2, y - tamaño / 2), tamaño, tamaño,
                    linewidth=0,  # Sin bordes
                    edgecolor='None',  # Sin contorno
                    facecolor=colores[raiz.tipo]
                )
                ax.add_patch(rect)

            # Dibujar los hijos recursivamente
            for cuadrante in raiz.cuadrantes:
                dibujar_aux(cuadrante, ax)

        # Crear un solo gráfico
        fig, ax = plt.subplots(figsize=(self.tamaño, self.tamaño))
        #fig.patch.set_facecolor('grey')

        # Configurar el área de dibujo
        ax.set_xlim(0, self.tamaño)
        ax.set_ylim(0, self.tamaño)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(str(self.semilla))

        # Dibujar el contenido
        dibujar_aux(self, ax)

        if self.caminos:
            caminos = self.caminos
            lado = 2 ** (self.profundidad + 1)
            for camino in caminos:
                dim = self.C.nodes[camino]['dimensiones']
                # Obtener parámetros como en un Cuadrado
                x = dim['centro'][0]
                y = dim['centro'][1]
                tamaño = dim['tamaño']
                # print(dim['centro'])
                # print(tamaño)
                # print("\n")
                # Dibujar igual que los cuadrantes
                rect = patches.Rectangle(
                    (x - tamaño / 2, y - tamaño / 2),
                    tamaño,
                    tamaño,
                    linewidth=0,
                    edgecolor='None',
                    facecolor=colores['camino']
                )
                ax.add_patch(rect)

        # Guardar la figura si es necesario
        if guardar:
            carpeta = "Mazmorras_generadas/Division_espacial"
            os.makedirs(carpeta, exist_ok=True)

            archivo_guardado = os.path.join(
                carpeta, f"DivEsp_{self.semilla}.png"
            )
            fig.savefig(archivo_guardado, transparent=False, bbox_inches='tight', dpi=300)
            #print(f"Figura guardada en: {archivo_guardado}")

    def mostrar_mapa_y_grafo(self, guardar=False):

        # Crear matriz base
        max_x = max(x for (x, y) in self.C.nodes)
        max_y = max(y for (x, y) in self.C.nodes)
        matriz = [[0 for _ in range(max_x + 1)] for _ in range(max_y + 1)]

        for (x, y), data in self.C.nodes(data=True):
            matriz[y][x] = data['value']  # 0: pared, 1: sala, 2: camino

        # Posiciones de los nodos en el grafo de salas
        pos = {}
        for idx, sala in enumerate(self.salas):
            celdas = [(x, y) for (x, y), d in self.C.nodes(data=True) if d['nodo'] in sala and d['value'] == 1]
            if celdas:
                xs, ys = zip(*celdas)
                centro = (sum(xs) / len(xs), sum(ys) / len(ys))
                pos[idx] = centro

        # Dibujar
        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = ListedColormap(['black', 'white', 'gray'])  # 0 = pared, 1 = sala, 2 = camino
        ax.imshow(matriz, cmap=cmap, origin='lower')

        # Dibujar grafo sobre el mapa
        nx.draw(self.grafo_salas, pos, with_labels=True, node_color='red',
                edge_color='yellow', node_size=400, ax=ax, font_size=10)

        # Etiquetas de peso (distancia)
        edge_labels = nx.get_edge_attributes(self.grafo_salas, 'weight')
        nx.draw_networkx_edge_labels(self.grafo_salas, pos,
                                     edge_labels={k: f"{v:.0f}" for k, v in edge_labels.items()},
                                     font_color='blue', ax=ax)

        ax.set_title("Mapa y grafo de salas superpuesto")
        ax.axis('off')

        if guardar:
            carpeta = "Mazmorras_generadas/Division_espacial"
            os.makedirs(carpeta, exist_ok=True)

            archivo_guardado = os.path.join(
                carpeta, f"DivEsp_{self.semilla}_grafo.png"
            )
            fig.savefig(archivo_guardado, transparent=False, bbox_inches='tight', dpi=300)
            #print(f"Figura guardada en: {archivo_guardado}")

        plt.show()

def crear_mazmorra_space_part(npart = 20, prof_division = 5, nsalas = 5, densidad = 1, guardar = False, semilla=None):
    if densidad > 3 or densidad < 0:
        print("La densidad debe estar entre 0 y 3")
        return
    A, semilla = formar_particion_estado_4(npart, d = prof_division, semilla=semilla)
    cuadrado = Cuadrado(A,semilla=semilla)
    #cuadrado.imprimir_estructura()
    cuadrado.formar_grafo()
    #cuadrado.dibujar_grafo()
    cuadrado.crear_mazmorra_clasica(nsalas, densidad=4-densidad)

    #cuadrado.dibujar_debug(guardar=guardar)
    cuadrado.dibujar(guardar=guardar)
    cuadrado.mostrar_mapa_y_grafo(guardar=guardar)

    m = cuadrado.obtener_matriz()
    g = cuadrado.grafo_salas
    return m,g
"""
------------------------------------------------------------------------------------------------------------------------
                                                    MÉTRICAS
------------------------------------------------------------------------------------------------------------------------
"""
def seleccionar_inicio_y_final(grafo):
    inicio = random.choice(list(grafo.nodes))
    distancias = nx.single_source_dijkstra_path_length(grafo, inicio)
    hojas = [n for n in grafo.nodes if grafo.degree[n] == 1]

    if hojas:
        final = max(hojas, key=lambda n: distancias.get(n, -1))
    else:
        final = max(distancias, key=distancias.get)

    return inicio, final

def calcular_metricas_topologicas(grafo):
    if not nx.is_connected(grafo):
        print("⚠️ El grafo no está completamente conectado. Se usará el componente más grande.")
        grafo = grafo.subgraph(max(nx.connected_components(grafo), key=len)).copy()

    # 1. Profundidad media (APL)
    apl = nx.average_shortest_path_length(grafo)

    # 2. Factor de ramificación
    total_nodos = grafo.number_of_nodes()
    nodos_hoja = sum(1 for n in grafo.nodes if grafo.degree[n] == 1)
    factor_ramificacion = 1 - (nodos_hoja / total_nodos)

    # 3. Media de betweenness
    betweenness = nx.betweenness_centrality(grafo, normalized=True)
    media_betweenness = statistics.mean(betweenness.values())

    # 4. Ratio de callejones
    ratio_callejones = nodos_hoja / total_nodos

    # 5. Decision density: nº de intersecciones / APL
    intersecciones = sum(1 for n in grafo.nodes if grafo.degree[n] > 2)
    decision_density = intersecciones / apl if apl > 0 else 0

    # Mostrar resultados
    # print("📊 Métricas topológicas del grafo:")
    # print(f"- Profundidad media (APL): {apl:.3f}")
    # print(f"- Factor de ramificación: {factor_ramificacion:.3f}")
    # print(f"- Media de betweenness: {media_betweenness:.5f}")
    # print(f"- Ratio de callejones (nodos hoja / totales): {ratio_callejones:.3f}")
    # print(f"- Decision density (intersecciones / APL): {decision_density:.3f}")

    return {
        "APL": apl,
        "Factor de ramificación": factor_ramificacion,
        "Media betweenness": media_betweenness,
        "Ratio de callejones": ratio_callejones,
        "Decision density": decision_density
    }

def calcular_metricas_espaciales(matriz):
    matriz = np.array(matriz)
    altura, anchura = matriz.shape

    # 1. Ratio suelo/pared
    suelo = (matriz > 0).sum()
    total = matriz.size
    ratio_suelo = suelo / total

    # 2. Compactación
    coords = np.argwhere(matriz > 0)

    # Si hay más de una celda, calculamos la varianza de sus coordenadas
    if len(coords) > 1:
        var_x = np.var(coords[:, 1])  # Varianza en columnas (eje X)
        var_y = np.var(coords[:, 0])  # Varianza en filas (eje Y)
        compactacion = (var_x + var_y) / 2  # Media de ambas
    else:
        compactacion = 0  # Si solo hay una celda o ninguna

    # 3. Rugosidad
    # Calculamos cuántos bordes expuestos tiene cada celda transitable
    borde = 0
    for x in range(altura):
        for y in range(anchura):
            if matriz[x, y] > 0:
                # Revisamos las 4 direcciones cardinales
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    # Si el vecino está fuera de la matriz o es pared
                    if nx < 0 or ny < 0 or nx >= altura or ny >= anchura or matriz[nx, ny] == 0:
                        borde += 1

    # Rugosidad = borde / área
    rugosidad = borde / suelo if suelo > 0 else 0

    # print("📐 Métricas de densidad y espacio:")
    # print(f"- Ratio suelo/pared: {ratio_suelo:.3f}")
    # print(f"- Compactación (varianza media): {compactacion:.2f}")
    # print(f"- Rugosidad (bordes por celda): {rugosidad:.2f}")

    return {
        "Ratio suelo/pared": ratio_suelo,
        "Compactación": compactacion,
        "Rugosidad": rugosidad
    }

def calcular_ritmo_exploracion(grafo, entrada, salida):
    if not nx.has_path(grafo, entrada, salida):
        print("No hay camino entre entrada y salida")
        return {
            "Longitud camino crítico": None,
            "Cantidad de caminos": 0
        }

    # 1. Longitud del camino crítico (camino más corto)
    camino_critico = nx.shortest_path_length(grafo, source=entrada, target=salida)

    # 2. Cantidad de caminos distintos (sin ciclos)
    caminos = list(nx.all_simple_paths(grafo, source=entrada, target=salida))
    cantidad_caminos = len(caminos)

    # Mostrar
    # print("⏳ Métricas de ritmo de exploración:")
    # print(f"- Longitud del camino crítico: {camino_critico}")
    # print(f"- Cantidad de caminos distintos: {cantidad_caminos}")

    return {
        "Longitud camino crítico": camino_critico,
        "Cantidad de caminos": cantidad_caminos
    }

"""
------------------------------------------------------------------------------------------------------------------------
                                                    EVALUACIÓN
------------------------------------------------------------------------------------------------------------------------
"""
def guardar_log_error(nombre_algoritmo, config, semilla, mensaje):
    log_dir = os.path.join("Resultados", nombre_algoritmo)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "errores.log")
    with open(log_path, "a") as log_file:
        log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
        log_file.write(f"Semilla: {semilla}\nConfig: {config}\nError: {mensaje}\n\n")
def generar_combinaciones(parametros_dict):
    """
    Dado un diccionario como:
        {"a": [1,2], "b": [3,4]}
    Devuelve:
        [{"a":1,"b":3}, {"a":1,"b":4}, {"a":2,"b":3}, {"a":2,"b":4}]
    """
    claves = list(parametros_dict.keys())
    valores = list(parametros_dict.values())
    combinaciones = list(product(*valores))
    return [dict(zip(claves, combinacion)) for combinacion in combinaciones]

def evaluar_algoritmo(nombre_algoritmo, generador_func, combinaciones_param, carpeta_salida):
    resultados = []
    os.makedirs(os.path.join("Resultados", carpeta_salida), exist_ok=True)

    tiempo_global_inicio = time.time()

    for config in generar_combinaciones(combinaciones_param):
        for _ in range(3):  # Tres semillas por combinación
            semilla = random.randint(0, 2**32 - 1)
            try:
                start_time = time.time()
                matriz, grafo = generador_func(semilla=semilla, guardar=True, **config)
                tiempo = time.time() - start_time

                metricas_topo = calcular_metricas_topologicas(grafo)
                metricas_esp = calcular_metricas_espaciales(matriz)

                longitudes = []
                cantidades = []

                for _ in range(5):
                    entrada, salida = seleccionar_inicio_y_final(grafo)
                    try:
                        res = calcular_ritmo_exploracion(grafo, entrada, salida)
                        longitudes.append(res.get("Longitud camino crítico", 0))
                        cantidades.append(res.get("Cantidad de caminos", 0))
                    except Exception:
                        continue  # en caso de error o camino imposible

                if longitudes:
                    metricas_ritmo = {
                        "Longitud camino crítico": sum(longitudes) / len(longitudes),
                        "Cantidad de caminos": sum(cantidades) / len(cantidades)
                    }
                else:
                    metricas_ritmo = {
                        "Longitud camino crítico": 0,
                        "Cantidad de caminos": 0
                    }

                fila = {
                    "semilla": semilla,
                    **config,
                    **metricas_topo,
                    **metricas_esp,
                    **metricas_ritmo,
                    "tiempo_ejecucion": tiempo,
                    "fallo": False
                }
                resultados.append(fila)

            except Exception as e:
                fila = {
                    "semilla": semilla,
                    **config,
                    "tiempo_ejecucion": 0,
                    "fallo": True
                }
                resultados.append(fila)
                guardar_log_error(nombre_algoritmo, config, semilla, traceback.format_exc())

    tiempo_global_final = time.time()-tiempo_global_inicio

    # Guardar CSV
    df = pd.DataFrame(resultados)
    output_path = os.path.join("Resultados", carpeta_salida, "metricas.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(nombre_algoritmo)
    print(f"Resultados guardados en {output_path}")
    print(f"La ejecución ha durado {tiempo_global_final}\n")


parametros_random = {
    "anchura":[60, 80, 100],
    "altura":[60, 80, 100],
    "max_salas":[10, 20, 30]
}

parametros_celular = {
    "iteraciones":[3, 4, 5],
    "dimensiones":[50, 80],
    "p":[0.40, 0.50, 0.60],
    "pared":[3, 4, 5],
    "cueva":[5, 6],
    "num_semillas":[5, 10, 15]
}

parametros_division = {
    "arquitectura": ["clasica"],
    "npart":[40, 60, 80, 120],
    "prof_division":[3, 4, 5],
    "nsalas":[10, 20, 30],
    "densidad":[1, 2, 3]
}

#evaluar_algoritmo("Colocacion_aleatoria", generar_dungeon_ColAl, parametros_random, "Colocacion_aleatoria")
#evaluar_algoritmo("Automata_Celular", automata_celular, parametros_celular, "Automata_Celular")
#evaluar_algoritmo("Division_espacial", crear_mazmorra_space_part, parametros_division, "Division_espacial")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generador de mazmorras para TFG")
    parser.add_argument("--algoritmo", choices=["Colocacion", "Automata", "Division"], required=True,
                        help="Algoritmo de generación: random | celular | division")

    # Comunes
    parser.add_argument("--semilla", type=int, default=None, help="Semilla aleatoria")

    # Parámetros para colocación aleatoria
    parser.add_argument("--anchura", type=int, default=60, help="Anchura de la matriz (random)")
    parser.add_argument("--altura", type=int, default=30, help="Altura de la matriz (random)")
    parser.add_argument("--max_salas", type=int, default=10, help="Número máximo de salas (random)")
    parser.add_argument("--intentos_por_sala", type=int, default=10, help="Intentos por sala (random)")

    # Parámetros para autómata celular
    parser.add_argument("--dimensiones", type=int, default=50, help="Tamaño del mapa cuadrado (celular)")
    parser.add_argument("--p", type=float, default=0.45, help="Probabilidad inicial de muro (celular)")
    parser.add_argument("--pared", type=int, default=4, help="Vecinos para mantener muro (celular)")
    parser.add_argument("--cueva", type=int, default=5, help="Vecinos para convertir en suelo (celular)")
    parser.add_argument("--iteraciones", type=int, default=4, help="Iteraciones del autómata (celular)")
    parser.add_argument("--semillas", type=int, default=5, help="Número de salas (celular)")

    # Parámetros para división espacial
    parser.add_argument("--npart", type=int, default=20, help="Número de nodos en la partición (division)")
    parser.add_argument("--profundidad", type=int, default=5, help="Profundidad máxima de partición (division)")
    parser.add_argument("--nsalas", type=int, default=5, help="Número de salas a crear (division)")
    parser.add_argument("--densidad", type=int, default=1, help="Densidad de adyacencia (0–3) (division)")

    args = parser.parse_args()

    if args.algoritmo == "Colocacion":
        generar_dungeon_ColAl(
            anchura=args.anchura,
            altura=args.altura,
            max_salas=args.max_salas,
            semilla=args.semilla,
            intentos_por_sala=args.intentos_por_sala,
        )

    elif args.algoritmo == "Automata":
        automata_celular(
            dimensiones=args.dimensiones,
            p=args.p,
            pared=args.pared,
            cueva=args.cueva,
            iteraciones=args.iteraciones,
            num_semillas=args.semillas,
            semilla=args.semilla,
        )

    elif args.algoritmo == "Division":
        crear_mazmorra_space_part(
            npart=args.npart,
            prof_division=args.profundidad,
            nsalas=args.nsalas,
            densidad=args.densidad,
            semilla=args.semilla
        )
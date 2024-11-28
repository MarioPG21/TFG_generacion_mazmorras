import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import random

import random

import random


def formar_particion_estado_4(n):
    lista_hojas = []

    G = nx.DiGraph()
    G.add_node(1, profundidad=0)
    G.graph['altura'] = 0

    lista_hojas.append(1)
    nodos = 1

    for i in range(1, n + 1):
        selec = random.choice(lista_hojas)
        # print("Nodo seleccionado: "+str(selec))
        profundidad = nx.shortest_path_length(G, source=1, target=selec) + 1

        if profundidad == 5:
            continue

        if profundidad > G.graph['altura']:
            G.graph['altura'] = profundidad

        G.add_node(nodos + 1, profundidad=profundidad)
        G.add_node(nodos + 2, profundidad=profundidad)
        G.add_node(nodos + 3, profundidad=profundidad)
        G.add_node(nodos + 4, profundidad=profundidad)
        G.add_edges_from([(selec, nodos + 1), (selec, nodos + 2), (selec, nodos + 3), (selec, nodos + 4)])
        # print("Añadidos hijos "+str(nodos+1)+" y "+str(nodos+2))
        lista_hojas.append(nodos + 1)
        lista_hojas.append(nodos + 2)
        lista_hojas.append(nodos + 3)
        lista_hojas.append(nodos + 4)
        lista_hojas.remove(selec)
        nodos += 4

    return G

def describir_arbol(G):
    def describir_arbol_aux(nodo, G, profundidad):
        hijos = list(G.successors(nodo))
        resultado = "\t" * profundidad + f"Nodo: {nodo} | Profundidad: {G.nodes[nodo]['profundidad']}\n"
        if hijos != []:
            for hijo in hijos:
                resultado += describir_arbol_aux(hijo, G, profundidad + 1)
        return resultado

    nodes = list(G.nodes)
    string = describir_arbol_aux(nodes[0], G, 0)
    print(string)


import matplotlib.patches as patches

class Cuadrado():
    def __init__(self, G, nodo=1, centro=(0, 0), tamaño=10):
        self.G=G
        self.S = nx.Graph()
        tipos = ['suelo','pared']
        self.nodo = nodo
        self.centro = centro
        self.tamaño = tamaño
        self.tipo = random.choice(tipos)

        self.profundidad = self.G.graph['altura']

        self.cuadrantes = []

        self.hijos = []

        hijos = list(self.G.successors(nodo))

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
                self.cuadrantes.append(Cuadrado(self.G, hijo, nuevo_centro, mitad_tamaño))

    def __str__(self):
        """
        Representación en texto del cuadrado.
        """
        return f"Cuadrado(nodo={self.nodo},tamaño={self.tamaño},tipo={self.tipo})"

    def imprimir_estructura(self, nivel=0):
        """
        Imprime la estructura jerárquica del cuadrado.
        """
        print(" " * nivel * 4 + str(self))
        for cuadrante in self.cuadrantes:
            cuadrante.imprimir_estructura(nivel + 1)

    def dibujar(self, ax):
        # Dibuja el cuadrado actual
        x, y = self.centro
        tamaño = self.tamaño
        if self.tipo == 'pared':
            rect = patches.Rectangle(
                (x - tamaño / 2, y - tamaño / 2), tamaño, tamaño,
                linewidth=1, facecolor='black'
            )
            ax.add_patch(rect)
        elif self.tipo == 'suelo':
            rect = patches.Rectangle(
                (x - tamaño / 2, y - tamaño / 2), tamaño, tamaño,
                linewidth=1, facecolor='white'
            )
            ax.add_patch(rect)

        # Dibujar los hijos
        for cuadrante in self.cuadrantes:
            cuadrante.dibujar(ax)


G = formar_particion_estado_4(40)
cuadrado = Cuadrado(G)
#cuadrado.imprimir_estructura()
#print(cuadrado.profundidad)
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.axis('off')

# Configuración del fondo transparente
fig.patch.set_alpha(0)  # Fondo de la figura transparente
ax.patch.set_alpha(0)   # Fondo del eje transparente

cuadrado.dibujar(ax)

fig.savefig('Cuadrados/cuadrado.png', transparent=True, bbox_inches='tight')
#plt.show()
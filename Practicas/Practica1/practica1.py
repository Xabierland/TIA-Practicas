# -*- coding: utf-8 -*-
"""
Nombre y apellidos: Xabier Gabiña Barañano
Asignatura: Tecnicas de Inteligencia Artificial
Fecha: 16/09/2024
Tarea: Practica 1 - Busqueda en Profundidad (DFS)

Descripción: Implementar la búsqueda en profundidad (DFS) para encontrar un nodo objetivo en un árbol.
"""
# Librerias
import random
from collections import deque

# Variables Globales

# Objetos
class Nodo:
    def __init__(self, contenido=None):
        """
        Inicializa una nueva instancia de la clase.

        Args:
            contenido (opcional): El contenido que se asignará a la instancia. 
                                  Puede ser de cualquier tipo. Por defecto es None.

        Atributos:
            contenido: Almacena el contenido proporcionado al inicializar la instancia.
            hijos: Lista que almacenará los hijos de la instancia.
            es_objetivo: Booleano que indica si la instancia es el objetivo. Por defecto es False.
        """
        self.contenido = contenido
        self.hijos = []
        self.es_objetivo = False
    
    def isGoalState(self):
        """
        Verifica si el estado actual es el estado objetivo.
        Un estado se considera objetivo si tanto 'izquierda' como 'derecha' son None.
        Returns:
            bool: True si 'izquierda' y 'derecha' son None, de lo contrario False.
        """
        return self.es_objetivo
    
    def setGoalState(self):
        """
        Establece el estado actual como estado objetivo.
        """
        self.es_objetivo = True
    
    def getSuccesor(self):
        """
        Retorna los hijos del nodo.
        Returns:
            list: Lista con los hijos del nodo.
        """
        return self.hijos
    
    def addSuccesor(self, hijo):
        """
        Agrega un hijo al nodo.
        Args:
            hijo (Nodo): Nodo hijo a agregar.
        """
        self.hijos.append(hijo)
    
# Funciones
def generar_arbol(n):
    """
    Partiendo de un nodo raiz genera un arbol aleatorio con un numero de nodos n
    Cada nodo tendra una letra como contenido siendo la primera la A hasta terminar, en caso de no haber suficientes letras se repetiran
    Los nodos se añadiran como hijos de un nodo aleatorio
    Hay 1/n probabilidades de que el nodo sea el nodo objetivo
    """
    letras = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    raiz = Nodo(contenido=letras[0])
    nodos = [raiz]
    
    for i in range(1, n):
        nuevo_nodo = Nodo(contenido=letras[i % len(letras)])
        padre = random.choice(nodos)
        padre.addSuccesor(nuevo_nodo)
        nodos.append(nuevo_nodo)
    
    objetivo = random.choice(nodos)
    objetivo.setGoalState()
    
    return raiz

def imprimir_arbol(nodo_raiz):
    """
    Imprime un árbol en niveles desde el nodo raíz.
    Args:
        nodo_raiz (Nodo): El nodo raíz del árbol a imprimir.
    Returns:
        None
    El árbol se imprime nivel por nivel, de izquierda a derecha.
    Notas:
        - Utiliza una cola (deque) para realizar un recorrido por niveles (BFS).
        - Cada nivel del árbol se imprime en una nueva línea.
    """
    if not nodo_raiz:
        return
    
    cola = deque([nodo_raiz])
    
    while cola:
        nivel = len(cola)
        while nivel > 0:
            nodo = cola.popleft()
            print(nodo.contenido, end=' ')
            for hijo in nodo.hijos:
                cola.append(hijo)
            nivel -= 1
        print()  # Nueva línea para el siguiente nivel
    

def dfs_v1(nodo_raiz):
    """
    Implementación de la búsqueda en profundidad (DFS) para encontrar un nodo objetivo en un árbol.
    Como resultado, se imprimira, no solo, el nodo objetivo, sino también el camino que lleva a él.
    Args:
        nodo_raiz (Nodo): El nodo raíz del árbol en el que se realizará la búsqueda.
    Returns:
        None
    """
    stack = [nodo_raiz] # Pila para almacenar los nodos a visitar
    visited = set()     # Conjunto para almacenar los nodos visitados
    path = []           # Lista para almacenar el camino al nodo objetivo
    
    while stack:    # Mientras haya elementos en el stack
        nodo_actual = stack.pop()   # Sacar el último elemento de la pila
        if nodo_actual in visited:  # Si el nodo actual no ha sido visitado
            continue
        visited.add(nodo_actual)   # Marcar el nodo actual como visitado
        if nodo_actual.isGoalState():  # Si el nodo actual es el objetivo
            break
        for hijo in reversed(nodo_actual.getSuccesor()): # Añadir los hijos del nodo actual a la pila
            stack.append(hijo)
    print("Nodo objetivo encontrado:", nodo_actual.contenido)

def dfs_final(nodo_raiz):
    stack = [nodo_raiz] # Pila para almacenar los nodos a visitar
    visited = set()     # Conjunto para almacenar los nodos visitados
    path = []           # Lista para almacenar el camino al nodo objetivo
    
    while stack:    # Mientras haya elementos en el stack
        nodo_actual = stack.pop()   # Sacar el último elemento de la pila
        if nodo_actual in visited:  # Si el nodo actual no ha sido visitado
            continue
        visited.add(nodo_actual)   # Marcar el nodo actual como visitado
        if nodo_actual.isGoalState():  # Si el nodo actual es el objetivo
            break
        for hijo in reversed(nodo_actual.getSuccesor()): # Añadir los hijos del nodo actual a la pila
            stack.append(hijo)
    print("Nodo objetivo encontrado:", nodo_actual.contenido)


# Main
if __name__ == "__main__":
    print("Generando arbol...")
    raiz=generar_arbol(17)
    print("Arbol generado:")
    imprimir_arbol(raiz)
    print("DFS v1:")
    dfs_v1(raiz)
    print("DFS final:")
    dfs_final(raiz)
    
    
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from abc import ABC, abstractmethod

import util


class SearchProblem(ABC):
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    @abstractmethod
    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    @abstractmethod
    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    @abstractmethod
    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    @abstractmethod
    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Implementacion del algoritmo de busqueda en profundidad.
    
    Args:
        problem (SearchProblem): Problema de busqueda
    Returns:
        list: Lista de acciones para llegar al objetivo
    """
    stack = util.Stack()  # Añadir el nodo inicial a la pila
    stack.push([problem.getStartState(), []])
    visited = set()     # Conjunto para almacenar los nodos visitados
    
    while not stack.isEmpty():    # Mientras haya elementos en el stack
        nodo_actual = stack.pop()   # Sacar el último elemento de la pila
        if problem.isGoalState(nodo_actual[0]):  # Si el nodo actual es el objetivo
            return nodo_actual[1]  # Devolver el camino
        if nodo_actual[0] not in visited:
            visited.add(nodo_actual[0])
            for estado, accion, costo in reversed(problem.getSuccessors(nodo_actual[0])): # Añadir los hijos del nodo actual a la pila
                camino = nodo_actual[1] + [accion]
                stack.push([estado, camino])
            


def breadthFirstSearch(problem):
    """
    Implementacion del algoritmo de busqueda en anchura.
    
    Args:
        problem (SearchProblem): Problema de busqueda
    Returns:
        list: Lista de acciones para llegar al objetivo
    """
    queue = util.Queue()  # Añadir el nodo inicial a la cola
    queue.push([problem.getStartState(), []])
    visited = set()     # Conjunto para almacenar los nodos visitados
    
    while not queue.isEmpty():    # Mientras haya elementos en la cola
        nodo_actual = queue.pop()   # Sacar el primer elemento de la cola
        if problem.isGoalState(nodo_actual[0]):  # Si el nodo actual es el objetivo
            return nodo_actual[1]  # Devolver el camino
        if nodo_actual[0] not in visited:
            visited.add(nodo_actual[0])
            for estado, accion, costo in problem.getSuccessors(nodo_actual[0]): # Añadir los hijos del nodo actual a la cola
                camino = nodo_actual[1] + [accion]
                queue.push([estado, camino])


def uniformCostSearch(problem):
    """
    Implementacion del algoritmo de busqueda de coste uniforme.
    
    Args:
        problem (SearchProblem): Problema de busqueda
    Returns:
        list: Lista de acciones para llegar al objetivo
    """
    queue = util.PriorityQueue()  # Añadir el nodo inicial a el heap
    queue.push([problem.getStartState(), [], 0], 0)
    visited = set()     # Conjunto para almacenar los nodos visitados
    
    while not queue.isEmpty():    # Mientras haya elementos en el stack
        nodo_actual = queue.pop()   # Sacar el último elemento de la pila
        if problem.isGoalState(nodo_actual[0]):  # Si el nodo actual es el objetivo
            return nodo_actual[1]  # Devolver el camino
        if nodo_actual[0] not in visited:
            visited.add(nodo_actual[0])
            for estado, accion, costo in problem.getSuccessors(nodo_actual[0]): # Añadir los hijos del nodo actual a la pila
                camino = nodo_actual[1] + [accion]
                queue.push([estado, camino, nodo_actual[2] + costo], nodo_actual[2] + costo)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Implementacion del algoritmo de busqueda A*.
    
    Args:
        problem (SearchProblem): Problema de busqueda
        heuristic (function): Heuristica para el problema
    Returns:
        list: Lista de acciones para llegar al objetivo
    """
    queue = util.PriorityQueue()  # Añadir el nodo inicial a el heap
    queue.push([problem.getStartState(), [], 0], 0)
    visited = set()     # Conjunto para almacenar los nodos visitados
    
    while not queue.isEmpty():    # Mientras haya elementos en el stack
        nodo_actual = queue.pop()   # Sacar el último elemento de la pila
        if problem.isGoalState(nodo_actual[0]):  # Si el nodo actual es el objetivo
            return nodo_actual[1]  # Devolver el camino
        if nodo_actual[0] not in visited:
            visited.add(nodo_actual[0])
            for estado, accion, costo in problem.getSuccessors(nodo_actual[0]): # Añadir los hijos del nodo actual a la pila
                camino = nodo_actual[1] + [accion]
                queue.push([estado, camino, nodo_actual[2] + costo], nodo_actual[2] + costo + heuristic(estado, problem))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

# multiAgents.py
# --------------
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


import random

import util
from game import Agent
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    Un agente reflexivo que elige acciones basándose en una función de evaluación.
    """

    def getAction(self, gameState):
        """
        Devuelve la mejor acción para Pacman basada en la función de evaluación.
        """
        # Obtiene las acciones legales para Pacman
        legalMoves = gameState.getLegalActions()

        # Calcula los puntajes para cada acción utilizando la función de evaluación
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)  # Encuentra el puntaje más alto
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Elige aleatoriamente entre las mejores acciones

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Calcula el valor del estado sucesor después de que Pacman toma la acción `action`.
        Devuelve un valor numérico mayor para estados más favorables.
        """
        # Generar el estado sucesor
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # Obtiene la posición de Pacman después de moverse
        newPos = successorGameState.getPacmanPosition()
        # Obtiene la matriz de comida en el estado sucesor
        newFood = successorGameState.getFood()
        # Obtiene la lista de estados de los fantasmas
        newGhostStates = successorGameState.getGhostStates()
        # Obtiene los tiempos restantes de los fantasmas asustados
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Inicializar el puntaje con el puntaje base del sucesor
        score = successorGameState.getScore()

        # 1. Distancia a la comida más cercana
        foodList = newFood.asList()  # Convertir la matriz de comida a una lista de posiciones
        if foodList:  # Si hay comida disponible
            # Calcular la distancia mínima a la comida más cercana
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
            # Invertir la distancia para que un menor valor de distancia dé un mayor puntaje
            score += 10.0 / minFoodDistance

        # 2. Distancia a los fantasmas no asustados
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            if ghostDistance < 2 and ghost.scaredTimer == 0:  # Fantasma no asustado y muy cerca
                score -= 1000  # Penalización fuerte por estar demasiado cerca de un fantasma peligroso

        # 3. Incentivo por acercarse a fantasmas asustados
        for i, ghost in enumerate(newGhostStates):
            if newScaredTimes[i] > 0:  # Si el fantasma está asustado
                ghostDistance = manhattanDistance(newPos, ghost.getPosition())
                score += 200.0 / (ghostDistance + 1)  # Premiar estar cerca de un fantasma asustado

        # 4. Penalización por comida restante
        score -= len(foodList) * 10  # Penalizar por cada comida restante en el estado sucesor

        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Agente que implementa el algoritmo Minimax.
    """

    def getAction(self, gameState):
        """
        Devuelve la mejor acción para Pacman desde el estado actual `gameState` usando Minimax.
        """
        # Llama a la función minimax empezando con el agente 0 (Pacman) y profundidad 0
        best_action, _ = self.minimax(gameState, agentIndex=0, depth=0)
        return best_action

    def minimax(self, gameState, agentIndex, depth):
        """
        Función minimax que devuelve la mejor acción y su valor para el agente actual.
        """
        # Si el estado es terminal (gana o pierde) o alcanzamos la profundidad máxima, evaluamos el estado
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return None, self.evaluationFunction(gameState)

        if agentIndex == 0: # Pacman - Maximizador
            return self.max_value(gameState, agentIndex, depth)
        else:               # Fantasmas - Minimizadores
            return self.min_value(gameState, agentIndex, depth)

    def max_value(self, gameState, agentIndex, depth):
        """
        Calcula el valor máximo para el agente Pacman (maximizador).
        """
        # Inicializar el mejor valor y la mejor acción
        best_value = float('-inf')
        best_action = None

        # Recorre todas las acciones legales para Pacman
        for action in gameState.getLegalActions(agentIndex):
            # Generar el estado sucesor
            successorState = gameState.generateSuccessor(agentIndex, action)
            # Calcular el valor del sucesor usando minimax con el siguiente agente
            _, successor_value = self.minimax(successorState, agentIndex + 1, depth)
            # Actualiza el valor máximo si se encuentra un mejor valor
            if successor_value > best_value:
                best_value = successor_value
                best_action = action

        return best_action, best_value

    def min_value(self, gameState, agentIndex, depth):
        """
        Calcula el valor mínimo para los fantasmas (minimizador).
        """
        # Inicializar el peor valor y la mejor acción
        worst_value = float('inf')
        best_action = None

        # Recorre todas las acciones legales para el fantasma actual
        for action in gameState.getLegalActions(agentIndex):
            # Generar el estado sucesor
            successorState = gameState.generateSuccessor(agentIndex, action)
            
            # Si es el último fantasma, pasamos a Pacman incrementando la profundidad
            if agentIndex == gameState.getNumAgents() - 1:
                # Calcula el valor del sucesor con Pacman y siguiente nivel de profundidad
                _, successor_value = self.minimax(successorState, 0, depth + 1)
            else:
                # Calcula el valor del sucesor con el siguiente fantasma
                _, successor_value = self.minimax(successorState, agentIndex + 1, depth)

            # Actualiza el valor mínimo si se encuentra un peor valor
            if successor_value < worst_value:
                worst_value = successor_value
                best_action = action

        return best_action, worst_value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pacman_pos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction

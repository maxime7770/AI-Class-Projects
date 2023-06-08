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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        capsules = successorGameState.getCapsules()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        distCapsule = -2 * min([manhattanDistance(newPos, capsule) for capsule in capsules]) if len(capsules) > 0 else 0
        distFood = -3 * min([manhattanDistance(newPos, food) for food in newFood.asList()]) if len(newFood.asList()) > 0 else 0
        distGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]) if len(newGhostStates) > 0 else 0 
        if distGhost < 5 and distGhost > 2:
            distGhost *= -40
        elif distGhost <= 2:
            distGhost *= -80
        diffFood = len(currentGameState.getFood().asList()) - 30 * len(newFood.asList())
        minScared = min(newScaredTimes)
        scared = 40 * minScared
        return distFood + distGhost + scared + diffFood + distCapsule + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minValue(self, state, depth, index):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        v = float("inf")
        for a in state.getLegalActions(index):
            if index == state.getNumAgents() - 1:
                v = min(v, self.maxValue(state.generateSuccessor(index, a), depth + 1, 0))
            else:
                v = min(v, self.minValue(state.generateSuccessor(index, a), depth, index + 1))
        return v

    def maxValue(self, state, depth, index):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        v = float("-inf")
        for a in state.getLegalActions(index):
            v = max(v, self.minValue(state.generateSuccessor(index, a),  depth, 1))
        return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        v = float("-inf")
        bestAction = None
        for a in gameState.getLegalActions(0):
            temp = self.minValue(gameState.generateSuccessor(0, a), 0, 1)
            if temp > v:
                v = temp
                bestAction = a
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def minValue(self, state, depth, index, alpha, beta):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        v = float("inf")
        for a in state.getLegalActions(index):
            if index == state.getNumAgents() - 1:
                v = min(v, self.maxValue(state.generateSuccessor(index, a), depth + 1, 0, alpha, beta))
            else:
                v = min(v, self.minValue(state.generateSuccessor(index, a), depth, index + 1, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def maxValue(self, state, depth, index, alpha, beta):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        v = float("-inf")
        for a in state.getLegalActions(index):
            v = max(v, self.minValue(state.generateSuccessor(index, a),  depth, 1, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        v = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        bestAction = None
        for a in gameState.getLegalActions(0):
            temp = self.minValue(gameState.generateSuccessor(0, a), 0, 1, alpha, beta)
            if temp > v:
                v = temp
                bestAction = a
            if v > beta:
                return bestAction
            alpha = max(alpha, v)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def meanValue(self, state, depth, index):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        v = 0
        for a in state.getLegalActions(index):
            if index == state.getNumAgents() - 1:
                v += self.maxValue(state.generateSuccessor(index, a), depth + 1, 0)
            else:
                v += self.meanValue(state.generateSuccessor(index, a), depth, index + 1)
        return v / len(state.getLegalActions(index))

    def maxValue(self, state, depth, index):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        v = float("-inf")
        for a in state.getLegalActions(index):
            v = max(v, self.meanValue(state.generateSuccessor(index, a),  depth, 1))
        return v

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        v = float("-inf")
        bestAction = None
        for a in gameState.getLegalActions(0):
            temp = self.meanValue(gameState.generateSuccessor(0, a), 0, 1)
            if temp > v:
                v = temp
                bestAction = a
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    The evaluation function is based on the distance to the nearest food, the distance to the nearest ghost, the number
    of food left, the number of capsules left, and the scared time of the ghosts. The closer the food is, the higher the
    score is. The closer the ghost is, the lower the score is. The more food left, the higher the score is. The more
    capsules left, the higher the score is. The longer the scared time of the ghosts, the higher the score is.
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    foodList = food.asList()
    foodNum = 5 / len(foodList) if len(foodList) > 0 else 0
    foodDist = [manhattanDistance(pos, f) for f in foodList]
    ghostDist = [manhattanDistance(pos, g.getPosition()) for g in ghostStates]
    capsuleDist = [manhattanDistance(pos, c) for c in capsules]
    foodScore = 3 / min(foodDist) if len(foodDist) > 0 else 0
    ghostScore = - min(ghostDist) if len(ghostDist) > 0 else 0
    capsuleScore = -10 * min(capsuleDist) if len(capsuleDist) > 0 else 0
    scaredScore = 10 * sum(scaredTimes)
    if ghostScore < 5 and ghostScore > 2:
        ghostScore *= 30
    elif ghostScore <= 2:
        ghostScore *= 40
    
    return currentGameState.getScore() + foodScore + ghostScore + capsuleScore + scaredScore + foodNum


# Abbreviation
better = betterEvaluationFunction

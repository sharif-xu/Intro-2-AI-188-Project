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
from game import Directions
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        evaluation = currentGameState.getScore()
        if successorGameState.isWin():
            return 1000
        if successorGameState.isLose():
            return -1000

        """ Hint from piazza: try some reciprocal here. """
        """ When food gets nearer, the value is greater. """
        """ When there are less food, the value is greater. """
        foodDist = 10000
        for food in newFood:
            foodDist = min(foodDist, manhattanDistance(newPos, food))
        # evaluation += float(1 / foodDist) * 10
        # evaluation -= float(1 / len(newFood.asList())) * 10
        evaluation += foodDist
        evaluation -= len(newFood.asList())

        """ Add some weight of ghost distance into evaluation. """
        """ Ghost moves. """
        """ When ghost gets closer, the value is smaller. """
        ghostDist = 10000
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            ghostDist = min(manhattanDistance(newPos, ghostPos), ghostDist)
        # evaluation -= float(1 / ghostDist) * 10
        evaluation -= ghostDist

        """ Keep pacman moving. """
        if action == Directions.STOP:
            evaluation -= 1


        return evaluation

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        def maxValue(gameState, depth):
            actions = gameState.getLegalActions(0)
            if not actions or gameState.isWin() or gameState.isLose() or depth == self.depth:
                # no legal actions or reach height
                return self.evaluationFunction(gameState)
            maxV = -1000
            for action in actions:
                maxV = max(minValue(gameState.generateSuccessor(0, action), 1, depth + 1), maxV)
            return maxV

        def minValue(gameState, agentIndex, depth):
            actions = gameState.getLegalActions(agentIndex)
            if not actions or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if agentIndex == gameState.getNumAgents() - 1:
                # pacman moves last in a round
                minV = 1000
                for action in actions:
                    minV = min(maxValue(gameState.generateSuccessor(agentIndex, action), depth), minV)
            else:
                minV = 1000
                for action in actions:
                    minV = min(minValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth), minV)
            return minV

        actions = []
        for action in gameState.getLegalActions(0):
            actions.append((action, minValue(gameState.generateSuccessor(0, action), 1, 1)))
        maxV = -1000
        tmpAction = actions[0][0]
        for action in actions:
            if action[1] > maxV:
                maxV = max(action[1], maxV)
                tmpAction = action[0]
        return tmpAction

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth, alpha, beta):
            actions = gameState.getLegalActions(0)
            if not actions or gameState.isWin() or gameState.isLose() or depth == self.depth:
                # no legal actions or reach height
                return self.evaluationFunction(gameState)

            cntAlpha = alpha
            maxV = -1000
            for action in actions:
                maxV = max(minValue(gameState.generateSuccessor(0, action), 1, depth + 1, cntAlpha, beta), maxV)
                if maxV > beta:
                    return maxV
                cntAlpha = max(cntAlpha, maxV)
            return maxV

        def minValue(gameState, agentIndex, depth, alpha, beta):
            actions = gameState.getLegalActions(agentIndex)
            if not actions or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            cntBeta = beta
            minV = 1000
            if agentIndex == gameState.getNumAgents() - 1:
                # pacman moves last in a round
                for action in actions:
                    minV = min(minV, maxValue(gameState.generateSuccessor(agentIndex, action), depth, alpha, cntBeta), minV)
                    if minV < alpha:
                        return minV
                    cntBeta = min(cntBeta, minV)
            else:
                for action in actions:
                    minV = min(minValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, cntBeta), minV)
                    if minV < alpha:
                        return minV
                    cntBeta = min(cntBeta, minV)
            return minV

        actions = []
        alpha = -1000
        beta = 1000
        for action in gameState.getLegalActions(0):
            value = minValue(gameState.generateSuccessor(0, action), 1, 1, alpha, beta)
            actions.append((action, value))
            if value > beta:
                return action
            alpha = max(alpha, value)
        maxV = -1000
        tmpAction = actions[0][0]
        for action in actions:
            if action[1] > maxV:
                maxV = max(action[1], maxV)
                tmpAction = action[0]
        return tmpAction

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
        def maxValue(gameState, depth):
            actions = gameState.getLegalActions(0)
            if not actions or gameState.isWin() or gameState.isLose() or depth == self.depth:
                # no legal actions or reach height
                return self.evaluationFunction(gameState)
            maxV = -1000
            for action in actions:
                cntMaxV = expectValue(gameState.generateSuccessor(0, action), 1, depth + 1)
                maxV = max(maxV, cntMaxV)
            return maxV

        def expectValue(gameState, agentIndex, depth):
            actions = gameState.getLegalActions(agentIndex)
            if not actions or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            expectV = 0
            probability = 1.0 / len(gameState.getLegalActions(agentIndex))

            for action in actions:
                if agentIndex == gameState.getNumAgents() - 1:
                    cntExpectValue = maxValue(gameState.generateSuccessor(agentIndex, action), depth)
                else:
                    cntExpectValue = expectValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
                expectV += cntExpectValue * probability
            return expectV

        actions = []
        for action in gameState.getLegalActions(0):
            actions.append((action, expectValue(gameState.generateSuccessor(0, action), 1, 1)))
        maxV = -1000
        tmpAction = actions[0][0]
        for action in actions:
            if action[1] > maxV:
                maxV = max(action[1], maxV)
                tmpAction = action[0]
        return tmpAction

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currGhost = currentGameState.getGhostStates()
    currCapsule = currentGameState.getCapsules()

    bEvaluation = 0
    if currentGameState.isWin():
        return 1000
    if currentGameState.isLose():
        return -1000

    foodDist = 10000
    for food in currFood.asList():
        foodDist = min(foodDist, manhattanDistance(currPos, food))
    bEvaluation += float(1 / foodDist) * 10
    bEvaluation -= float(1 / len(currFood.asList()))

    ghostDist = 10000
    for ghost in currGhost:
        ghostPos = ghost.getPosition()
        ghostDist = min(manhattanDistance(currPos, ghostPos), ghostDist)
    bEvaluation -= float(1 / ghostDist) * 10

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

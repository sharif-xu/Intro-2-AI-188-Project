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

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

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


class Node(object):
    def __init__(self, location, direction, path):
        self.location = location
        self.direction = direction
        self.path = path


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    visited = set()
    stack = util.Stack()
    node = Node(problem.getStartState(), [], [])
    stack.push(node)

    if problem.isGoalState(node.location):
        return node.path

    while True:
        current_node = stack.pop()

        if problem.isGoalState(current_node.location):
            return current_node.path

        next_node_arrays = problem.getSuccessors(current_node.location)

        if current_node.location in visited:
            continue
        else:
            visited.add(current_node.location)


        for nextNode in next_node_arrays:
            if nextNode[0] not in visited:
                next_node = Node(nextNode[0], nextNode[1], current_node.path + [nextNode[1]])
                stack.push(next_node)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    visited = set()
    queue = util.Queue() #change Stack to Queue
    node = Node(problem.getStartState(), [], [])
    queue.push(node)

    if problem.isGoalState(node.location):
        return node.path

    while True:
        current_node = queue.pop()

        if problem.isGoalState(current_node.location):
            return current_node.path

        if current_node.location in visited:
            continue
        else:
            visited.add(current_node.location)
            next_node_arrays = problem.getSuccessors(current_node.location)

        for nextNode in next_node_arrays:
            if nextNode[0] not in visited:
                next_node = Node(nextNode[0], nextNode[1], current_node.path + [nextNode[1]])
                queue.push(next_node)

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    path = []
    value = 0
    visited = set()
    pri_queue = util.PriorityQueue()
    start = problem.getStartState()
    pri_queue.push((start, path, value), 0)

    while True:
        current_node, path, value = pri_queue.pop()
        if problem.isGoalState(current_node):
            break
        # Goal not found, continue the loop
        if current_node not in visited:
            visited.add(current_node)
            sucs_arr = problem.getSuccessors(current_node)

            for suc in sucs_arr:
                suc_node = suc[0]
                suc_path = suc[1]
                suc_value = suc[2]
                if suc_node not in visited:
                    new_path = path + [suc_path]  # current path to the successor
                    new_value = value + suc_value
                    pri_queue.update((suc_node, new_path, new_value), new_value)

    return path

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    path = []
    value = 0
    visited = set()
    pri_queue = util.PriorityQueue()
    start = problem.getStartState()
    pri_queue.push((start, path, value), 0)

    while True:
        current_node, path, value = pri_queue.pop()
        if problem.isGoalState(current_node):
            break
        # Goal not found, continue the loop
        if current_node not in visited:
            visited.add(current_node)
            sucs_arr = problem.getSuccessors(current_node)

            for suc in sucs_arr:
                suc_node = suc[0]
                suc_path = suc[1]
                suc_value = suc[2]
                h_n = heuristic(suc_node, problem)
                if suc_node not in visited:
                    new_path = path + [suc_path]  # current path to the successor
                    new_value = value + suc_value
                    pri_queue.update((suc_node, new_path, new_value), new_value + h_n)

    return path

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

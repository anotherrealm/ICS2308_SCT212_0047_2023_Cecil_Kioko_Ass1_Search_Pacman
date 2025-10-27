# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
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
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first.
  Your search algorithm needs to return a list of actions that reaches
  the goal. Make sure to implement a graph search algorithm.
  """
  from util import Stack

  stack = Stack()
  visited = set()

  start_state = problem.getStartState()
  stack.push((start_state, []))

  while not stack.isEmpty():
      state, path = stack.pop()

      if problem.isGoalState(state):
          return path

      if state not in visited:
          visited.add(state)
          for successor, action, stepCost in problem.getSuccessors(state):
              if successor not in visited:
                  stack.push((successor, path + [action]))

  return []

def breadthFirstSearch(problem):
  """
  Search the shallowest nodes in the search tree first.
  """
  from util import Queue

  queue = Queue()
  visited = set()

  start_state = problem.getStartState()
  queue.push((start_state, []))

  while not queue.isEmpty():
      state, path = queue.pop()

      if problem.isGoalState(state):
          return path

      if state not in visited:
          visited.add(state)
          for successor, action, stepCost in problem.getSuccessors(state):
              if successor not in visited:
                  queue.push((successor, path + [action]))

  return []

def uniformCostSearch(problem):
  "Search the node of least total cost first."
  from util import PriorityQueue

  pq = PriorityQueue()
  visited = {}

  start_state = problem.getStartState()
  pq.push((start_state, [], 0), 0)

  while not pq.isEmpty():
      state, path, cost = pq.pop()

      if problem.isGoalState(state):
          return path

      if state not in visited or cost < visited[state]:
          visited[state] = cost
          for successor, action, stepCost in problem.getSuccessors(state):
              new_cost = cost + stepCost
              if successor not in visited or new_cost < visited.get(successor, float('inf')):
                  pq.push((successor, path + [action], new_cost), new_cost)

  return []

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  from util import PriorityQueue

  pq = PriorityQueue()
  visited = {}

  start_state = problem.getStartState()
  start_heuristic = heuristic(start_state, problem)
  pq.push((start_state, [], 0), start_heuristic)

  while not pq.isEmpty():
      state, path, cost = pq.pop()

      if problem.isGoalState(state):
          return path

      if state not in visited or cost < visited[state]:
          visited[state] = cost
          for successor, action, stepCost in problem.getSuccessors(state):
              new_cost = cost + stepCost
              if successor not in visited or new_cost < visited.get(successor, float('inf')):
                  priority = new_cost + heuristic(successor, problem)
                  pq.push((successor, path + [action], new_cost), priority)

  return []
    
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
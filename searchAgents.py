# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  
To select an agent, use the '-p' option when running pacman.py.  
Arguments can be passed to your agent using '-a'.  

Example:
    python pacman.py -p SearchAgent -a fn=depthFirstSearch
"""

from game import Directions, Agent, Actions
import util
import time
import search


class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP


#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    A general search agent that finds a path using a given search algorithm
    for a given search problem.
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        if fn not in dir(search):
            raise AttributeError(f"{fn} is not a search function in search.py.")
        func = getattr(search, fn)

        if 'heuristic' not in func.__code__.co_varnames:
            print(f'[SearchAgent] using function {fn}')
            self.searchFunction = func
        else:
            if heuristic in globals():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(f"{heuristic} is not a function in searchAgents.py or search.py.")
            print(f'[SearchAgent] using function {fn} and heuristic {heuristic}')
            self.searchFunction = lambda x: func(x, heuristic=heur)

        if prob not in globals() or not prob.endswith('Problem'):
            raise AttributeError(f"{prob} is not a search problem type in SearchAgents.py.")
        self.searchType = globals()[prob]
        print(f'[SearchAgent] using problem type {prob}')

    def registerInitialState(self, state):
        if self.searchFunction is None:
            raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state)
        self.actions = self.searchFunction(problem)
        totalCost = problem.getCostOfActions(self.actions)
        print(f'Path found with total cost of {totalCost} in {time.time() - starttime:.1f} seconds')
        if hasattr(problem, "_expanded"):
            print(f'Search nodes expanded: {problem._expanded}')

    def getAction(self, state):
        if not hasattr(self, 'actionIndex'):
            self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class PositionSearchProblem(search.SearchProblem):
    """
    A search problem for finding a path to a particular point on the Pacman board.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=True):
        self.walls = gameState.getWalls()
        self.startState = start or gameState.getPacmanPosition()
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if hasattr(__main__, "_display") and hasattr(__main__._display, "drawExpandedCells"):
                __main__._display.drawExpandedCells(self._visitedlist)
        return isGoal

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                cost = self.costFn((nextx, nexty))
                successors.append(((nextx, nexty), action, cost))
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)
        return successors

    def getCostOfActions(self, actions):
        if actions is None:
            return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += self.costFn((x, y))
        return cost


class StayEastSearchAgent(SearchAgent):
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 0.5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)


class StayWestSearchAgent(SearchAgent):
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)


def manhattanHeuristic(position, problem, info={}):
    xy1, xy2 = position, problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclideanHeuristic(position, problem, info={}):
    xy1, xy2 = position, problem.goal
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5


#####################################################
# Corners Problem and Heuristic
#####################################################

class CornersProblem(search.SearchProblem):
    def __init__(self, startingGameState):
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print(f'Warning: no food in corner {corner}')
        self._expanded = 0

        self.cornersVisited = []
        for corner in self.corners:
            if self.startingPosition == corner:
                self.cornersVisited.append((corner, True))
            else:
                self.cornersVisited.append((corner, False))
        self.cornersVisited = tuple(self.cornersVisited)

    def getStartState(self):
        return (self.startingPosition, self.cornersVisited)

    def isGoalState(self, state):
        return all(corner[1] for corner in state[1])

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]
            cornerSt = state[1]
            new_CornerSt = []
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextSt = (nextx, nexty)
                for corner in cornerSt:
                    pos = corner[0]
                    if nextSt == pos:
                        new_CornerSt.append((pos, True))
                    else:
                        new_CornerSt.append((pos, corner[1]))
                successors.append(((nextSt, tuple(new_CornerSt)), action, 1))
        self._expanded += 1
        return successors

    def getCostOfActions(self, actions):
        if actions is None:
            return 999999
        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    pos, cornerSt = state[0], state[1]
    dist = 0
    for corner in cornerSt:
        cPos = corner[0]
        if not corner[1]:
            dist = max(dist, abs(cPos[0] - pos[0]) + abs(cPos[1] - pos[1]))
    return dist


class AStarCornersAgent(SearchAgent):
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem


#####################################################
# Food Search Problem and Heuristic
#####################################################

class FoodSearchProblem:
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0
        self.heuristicInfo = {}

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        successors = []
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append((((nextx, nexty), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions):
        if actions is None:
            return 999999
        x, y = self.getStartState()[0]
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
        return len(actions)


class AStarFoodSearchAgent(SearchAgent):
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem


def foodHeuristic(state, problem):
    pos, food = state
    foodpos = food.asList()
    if len(foodpos) == 0:
        return 0
    fcost = [mazeDistance(pos, f, problem.startingGameState) for f in foodpos]
    return max(fcost)


#####################################################
# Closest Dot Search
#####################################################

class ClosestDotSearchAgent(SearchAgent):
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while currentState.getFood().count() > 0:
            nextPathSegment = self.findPathToClosestDot(currentState)
            self.actions += nextPathSegment
            for action in nextPathSegment:
                if action not in currentState.getLegalActions():
                    raise Exception(f'findPathToClosestDot returned an illegal move: {action}')
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print(f'Path found with cost {len(self.actions)}.')

    def findPathToClosestDot(self, gameState):
        problem = AnyFoodSearchProblem(gameState)
        actions = search.bfs(problem)
        return actions


class AnyFoodSearchProblem(PositionSearchProblem):
    def __init__(self, gameState):
        self.food = gameState.getFood()
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        return state in self.food.asList()


#####################################################
# Maze Distance Helper
#####################################################

def mazeDistance(point1, point2, gameState):
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], f'point1 is a wall: {point1}'
    assert not walls[x2][y2], f'point2 is a wall: {point2}'
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))

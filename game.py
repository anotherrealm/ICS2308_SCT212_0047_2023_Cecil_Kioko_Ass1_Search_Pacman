# game.py
# -------
# Licensing Information: You are free to use or extend these projects for educational purposes.
# The Pacman AI projects were developed at UC Berkeley.
# For more info: http://inst.eecs.berkeley.edu/~cs188/

from util import *
import time
import sys
import traceback
import threading
import signal
from io import StringIO

# Try to import boinc; otherwise provide a dummy
try:
    import importlib
    boinc = importlib.import_module('boinc')
    _BOINC_ENABLED = True
except Exception:
    class _BoincDummy:
        @staticmethod
        def set_fraction_done(fraction):
            return None
    boinc = _BoincDummy()
    _BOINC_ENABLED = False

#######################
# Timeout helper
#######################

class TimeoutFunctionException(Exception):
    pass

class TimeoutFunction:
    """
    Wraps a function call with a timeout. Uses threading to enforce a wall-clock timeout.
    When the function doesn't return in time, TimeoutFunctionException is raised in the wrapper.
    This mirrors the behavior used in many Berkeley distributions.
    """

    def __init__(self, function, timeout):
        self.function = function
        self.timeout = timeout

    def _target(self, args, kwargs):
        try:
            self._result = self.function(*args, **kwargs)
            self._exception = None
        except Exception as e:
            self._result = None
            self._exception = e

    def __call__(self, *args, **kwargs):
        if self.timeout is None or self.timeout <= 0:
            # No timeout requested; just call the function
            return self.function(*args, **kwargs)

        self._result = None
        self._exception = None
        thread = threading.Thread(target=self._target, args=(args, kwargs))
        thread.daemon = True
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive():
            # Can't forcibly kill the thread in Python safely; raise timeout in caller
            raise TimeoutFunctionException("Timed out after %d seconds" % self.timeout)
        if self._exception:
            # Re-raise the underlying exception
            raise self._exception
        return self._result

#######################
# Basic structures
#######################

class Agent:
    """
    Abstract agent: subclasses must implement getAction(self, state).
    """

    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        raiseNotDefined()

class Directions:
    NORTH = 'North'
    SOUTH = 'South'
    EAST  = 'East'
    WEST  = 'West'
    STOP  = 'Stop'

    LEFT =  {NORTH: WEST, SOUTH: EAST, EAST: NORTH, WEST: SOUTH, STOP: STOP}
    RIGHT = dict([(y, x) for x, y in LEFT.items()])
    REVERSE = {NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST, STOP: STOP}

class Configuration:
    """
    A Configuration tracks a (x,y) position and a direction.
    """

    def __init__(self, pos, direction):
        self.pos = pos
        self.direction = direction

    def getPosition(self):
        return self.pos

    def getDirection(self):
        return self.direction

    def isInteger(self):
        x, y = self.pos
        return x == int(x) and y == int(y)

    def __eq__(self, other):
        if other is None: return False
        return (self.pos == other.pos and self.direction == other.direction)

    def __hash__(self):
        return hash((self.pos, self.direction))

    def __str__(self):
        return "(x,y)=%s, %s" % (str(self.pos), str(self.direction))

    def generateSuccessor(self, vector):
        """
        Returns a new Configuration moved by 'vector' (dx,dy). Direction follows the vector.
        """
        x, y = self.pos
        dx, dy = vector
        direction = Actions.vectorToDirection(vector)
        if direction == Directions.STOP:
            direction = self.direction
        return Configuration((x + dx, y + dy), direction)

class AgentState:
    def __init__(self, startConfiguration, isPacman):
        self.start = startConfiguration
        self.configuration = startConfiguration
        self.isPacman = isPacman
        self.scaredTimer = 0

    def __str__(self):
        if self.isPacman:
            return "Pacman: " + str(self.configuration)
        else:
            return "Ghost: " + str(self.configuration)

    def __eq__(self, other):
        if other is None:
            return False
        return self.configuration == other.configuration and self.scaredTimer == other.scaredTimer

    def __hash__(self):
        return hash((self.configuration, self.scaredTimer))

    def copy(self):
        state = AgentState(self.start, self.isPacman)
        state.configuration = self.configuration
        state.scaredTimer = self.scaredTimer
        return state

    def getPosition(self):
        if self.configuration is None: return None
        return self.configuration.getPosition()

    def getDirection(self):
        return self.configuration.getDirection()

class Grid:
    """
    Simple boolean grid, backed by a list of lists.
    Access via grid[x][y] where origin (0,0) is lower-left.
    """

    def __init__(self, width, height, initialValue=False, bitRepresentation=None):
        if initialValue not in [False, True]:
            raise Exception("Grids can only contain booleans")
        self.width = width
        self.height = height
        self.data = [[initialValue for y in range(height)] for x in range(width)]
        if bitRepresentation:
            self._unpackBits(bitRepresentation)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __str__(self):
        out = [[str(self.data[x][y])[0] for x in range(self.width)] for y in range(self.height)]
        out.reverse()
        return '\n'.join([''.join(x) for x in out])

    def __eq__(self, other):
        if other is None: return False
        return self.data == other.data

    def __hash__(self):
        base = 1
        h = 0
        for l in self.data:
            for i in l:
                if i:
                    h += base
                base *= 2
        return hash(h)

    def copy(self):
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def count(self, item=True):
        return sum([x.count(item) for x in self.data])

    def asList(self, key=True):
        lst = []
        for x in range(self.width):
            for y in range(self.height):
                if self[x][y] == key:
                    lst.append((x, y))
        return lst

    # Bit-packing helpers used by Berkeley code; included for compatibility
    CELLS_PER_INT = 30

    def packBits(self):
        bits = [self.width, self.height]
        currentInt = 0
        for i in range(self.height * self.width):
            bit = self.CELLS_PER_INT - (i % self.CELLS_PER_INT) - 1
            x, y = self._cellIndexToPosition(i)
            if self[x][y]:
                currentInt += 2 ** bit
            if (i + 1) % self.CELLS_PER_INT == 0:
                bits.append(currentInt)
                currentInt = 0
        bits.append(currentInt)
        return tuple(bits)

    def _cellIndexToPosition(self, index):
        x = index // self.height
        y = index % self.height
        return x, y

    def _unpackBits(self, bits):
        cell = 0
        for packed in bits:
            for bit in self._unpackInt(packed, self.CELLS_PER_INT):
                if cell == self.width * self.height: break
                x, y = self._cellIndexToPosition(cell)
                self[x][y] = bit
                cell += 1

    def _unpackInt(self, packed, size):
        bools = []
        if packed < 0:
            raise ValueError("must be a positive integer")
        for i in range(size):
            n = 2 ** (self.CELLS_PER_INT - i - 1)
            if packed >= n:
                bools.append(True)
                packed -= n
            else:
                bools.append(False)
        return bools

def reconstituteGrid(bitRep):
    if type(bitRep) is not tuple:
        return bitRep
    width, height = bitRep[:2]
    return Grid(width, height, bitRepresentation=bitRep[2:])

#######################
# Actions helpers
#######################

class Actions:
    """
    Collection of static methods for actions.
    """
    _directions = {
        Directions.NORTH: (0, 1),
        Directions.SOUTH: (0, -1),
        Directions.EAST:  (1, 0),
        Directions.WEST:  (-1, 0),
        Directions.STOP:  (0, 0)
    }

    _directionsAsList = list(_directions.items())

    TOLERANCE = .001

    @staticmethod
    def reverseDirection(action):
        if action == Directions.NORTH: return Directions.SOUTH
        if action == Directions.SOUTH: return Directions.NORTH
        if action == Directions.EAST:  return Directions.WEST
        if action == Directions.WEST:  return Directions.EAST
        return action

    @staticmethod
    def vectorToDirection(vector):
        dx, dy = vector
        if dy > 0: return Directions.NORTH
        if dy < 0: return Directions.SOUTH
        if dx < 0: return Directions.WEST
        if dx > 0: return Directions.EAST
        return Directions.STOP

    @staticmethod
    def directionToVector(direction, speed=1.0):
        dx, dy = Actions._directions[direction]
        return dx * speed, dy * speed

    @staticmethod
    def getPossibleActions(config, walls):
        possible = []
        x, y = config.pos
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        # If between grid points, must continue in current direction
        if (abs(x - x_int) + abs(y - y_int) > Actions.TOLERANCE):
            return [config.getDirection()]

        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            next_y = y_int + dy
            # bounds check
            if next_x < 0 or next_x >= walls.width or next_y < 0 or next_y >= walls.height:
                continue
            if not walls[next_x][next_y]:
                possible.append(dir)
        return possible

    @staticmethod
    def getLegalNeighbors(position, walls):
        x, y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        neighbors = []
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            next_y = y_int + dy
            if next_x < 0 or next_x == walls.width: continue
            if next_y < 0 or next_y == walls.height: continue
            if not walls[next_x][next_y]: neighbors.append((next_x, next_y))
        return neighbors

    @staticmethod
    def getSuccessor(position, action):
        dx, dy = Actions.directionToVector(action)
        x, y = position
        return (x + dx, y + dy)

#######################
# GameStateData
#######################

class GameStateData:
    def __init__(self, prevState=None):
        if prevState != None:
            # copy basic fields
            self.food = prevState.food.shallowCopy() if hasattr(prevState.food, 'shallowCopy') else prevState.food
            self.capsules = prevState.capsules[:]
            self.agentStates = self.copyAgentStates(prevState.agentStates)
            self.layout = prevState.layout
            self._eaten = prevState._eaten[:]
            self.score = prevState.score
        else:
            self.food = None
            self.capsules = []
            self.agentStates = []
            self.layout = None
            self._eaten = []
            self.score = 0
        self._foodEaten = None
        self._capsuleEaten = None
        self._agentMoved = None
        self._lose = False
        self._win = False
        self.scoreChange = 0

    def deepCopy(self):
        state = GameStateData(self)
        state.food = self.food.deepCopy() if hasattr(self.food, 'deepCopy') else self.food
        state.layout = self.layout.deepCopy() if hasattr(self.layout, 'deepCopy') else self.layout
        state._agentMoved = self._agentMoved
        state._foodEaten = self._foodEaten
        state._capsuleEaten = self._capsuleEaten
        return state

    def copyAgentStates(self, agentStates):
        copiedStates = []
        for agentState in agentStates:
            copiedStates.append(agentState.copy())
        return copiedStates

    def __eq__(self, other):
        if other is None: return False
        if not self.agentStates == other.agentStates: return False
        if not self.food == other.food: return False
        if not self.capsules == other.capsules: return False
        if not self.score == other.score: return False
        return True

    def __hash__(self):
        return int((hash(tuple(self.agentStates)) + 13 * hash(self.food) + 113 * hash(tuple(self.capsules)) + 7 * hash(self.score)) % 1048575)

    def __str__(self):
        width, height = self.layout.width, self.layout.height
        map = Grid(width, height)
        if type(self.food) == type((1,2)):
            self.food = reconstituteGrid(self.food)
        for x in range(width):
            for y in range(height):
                food, walls = self.food, self.layout.walls
                map[x][y] = self._foodWallStr(food[x][y], walls[x][y])

        for agentState in self.agentStates:
            if agentState == None: continue
            if agentState.configuration == None: continue
            x, y = [int(i) for i in nearestPoint(agentState.configuration.pos)]
            agent_dir = agentState.configuration.direction
            if agentState.isPacman:
                map[x][y] = self._pacStr(agent_dir)
            else:
                map[x][y] = self._ghostStr(agent_dir)

        for x, y in self.capsules:
            map[x][y] = 'o'

        return str(map) + ("\nScore: %d\n" % self.score)

    def _foodWallStr(self, hasFood, hasWall):
        if hasFood:
            return '.'
        elif hasWall:
            return '%'
        else:
            return ' '

    def _pacStr(self, dir):
        if dir == Directions.NORTH:
            return 'v'
        if dir == Directions.SOUTH:
            return '^'
        if dir == Directions.WEST:
            return '>'
        return '<'

    def _ghostStr(self, dir):
        return 'G'  # simplified for text-mode

    def initialize(self, layout, numGhostAgents):
        self.food = layout.food.copy()
        self.capsules = layout.capsules[:]
        self.layout = layout
        self.score = 0
        self.scoreChange = 0

        self.agentStates = []
        numGhosts = 0
        for isPacman, pos in layout.agentPositions:
            if not isPacman:
                if numGhosts == numGhostAgents: continue
                else:
                    numGhosts += 1
            self.agentStates.append( AgentState(Configuration(pos, Directions.STOP), isPacman) )
        self._eaten = [False for a in self.agentStates]

#######################
# Game core
#######################

class Game:
    """
    The Game coordinates the running of a Pacman game.
    """

    OLD_STDOUT = None
    OLD_STDERR = None

    def __init__(self, agents, display, rules, startingIndex=0, muteAgents=False, catchExceptions=False):
        self.agentCrashed = False
        self.agents = agents
        self.display = display
        self.rules = rules
        self.startingIndex = startingIndex
        self.gameOver = False
        self.muteAgents = muteAgents
        self.catchExceptions = catchExceptions
        self.moveHistory = []
        self.totalAgentTimes = [0 for agent in agents]
        self.totalAgentTimeWarnings = [0 for agent in agents]
        self.agentTimeout = False

        # Prepare agent output buffers (StringIO objects) for capturing stdout/stderr per agent
        # Use io.StringIO class directly
        self.agentOutput = [StringIO() for agent in agents]

    def getProgress(self):
        if self.gameOver:
            return 1.0
        else:
            try:
                return self.rules.getProgress(self)
            except Exception:
                return 0.0

    def _agentCrash(self, agentIndex, quiet=False):
        "Helper for handling agent crashes"
        if not quiet:
            traceback.print_exc()
        self.gameOver = True
        self.agentCrashed = True
        try:
            self.rules.agentCrash(self, agentIndex)
        except Exception:
            pass

    def mute(self, agentIndex: int) -> None:
        """
        Redirect stdout/stderr for the given agent index to their own StringIO buffer.
        """
        if not self.muteAgents:
            return
        Game.OLD_STDOUT = sys.stdout
        Game.OLD_STDERR = sys.stderr
        sys.stdout = self.agentOutput[agentIndex]
        sys.stderr = self.agentOutput[agentIndex]

    def unmute(self) -> None:
        """
        Restore stdout/stderr to original values.
        """
        if not self.muteAgents:
            return
        if Game.OLD_STDOUT is not None:
            sys.stdout = Game.OLD_STDOUT
        else:
            sys.stdout = sys.__stdout__
        if Game.OLD_STDERR is not None:
            sys.stderr = Game.OLD_STDERR
        else:
            sys.stderr = sys.__stderr__

    def run(self):
        """
        Main control loop for game play.
        This method is intentionally robust: it times agent setup, enforces timeouts,
        updates display, calls rules, and gracefully handles exceptions.
        """
        # Initialize display with current state if present
        try:
            if hasattr(self, 'state') and self.state is not None and self.display is not None:
                self.display.initialize(self.state.data)
        except Exception:
            # Some displays may not implement initialize the same way; ignore
            pass

        self.numMoves = 0

        # Inform agents of the initial state via registerInitialState (if present)
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if not agent:
                # Null agent means it failed to load
                self.mute(i)
                print("Agent %d failed to load" % i)
                self.unmute()
                self._agentCrash(i, quiet=True)
                return
            if ("registerInitialState" in dir(agent)):
                self.mute(i)
                if self.catchExceptions:
                    try:
                        timed_func = TimeoutFunction(agent.registerInitialState, int(self.rules.getMaxStartupTime(i)))
                        try:
                            start_time = time.time()
                            timed_func(self.state.deepCopy())
                            time_taken = time.time() - start_time
                            self.totalAgentTimes[i] += time_taken
                        except TimeoutFunctionException:
                            print("Agent %d ran out of time on startup!" % i)
                            self.unmute()
                            self.agentTimeout = True
                            self._agentCrash(i, quiet=True)
                            return
                    except Exception:
                        self._agentCrash(i, quiet=False)
                        self.unmute()
                        return
                else:
                    try:
                        agent.registerInitialState(self.state.deepCopy())
                    except Exception:
                        self._agentCrash(i, quiet=False)
                        self.unmute()
                        return
                self.unmute()

        agentIndex = self.startingIndex
        numAgents = len(self.agents)

        # Main loop
        while not self.gameOver:
            agent = self.agents[agentIndex]
            move_time = 0
            skip_action = False

            # Generate observation
            if 'observationFunction' in dir(agent):
                self.mute(agentIndex)
                if self.catchExceptions:
                    try:
                        timed_func = TimeoutFunction(agent.observationFunction, int(self.rules.getMoveTimeout(agentIndex)))
                        try:
                            start_time = time.time()
                            observation = timed_func(self.state.deepCopy())
                        except TimeoutFunctionException:
                            skip_action = True
                        move_time += time.time() - start_time
                        self.unmute()
                    except Exception:
                        self._agentCrash(agentIndex, quiet=False)
                        self.unmute()
                        return
                else:
                    observation = agent.observationFunction(self.state.deepCopy())
                self.unmute()
            else:
                observation = self.state.deepCopy()

            # Solicit an action from the agent
            action = None
            self.mute(agentIndex)
            if self.catchExceptions:
                try:
                    timed_func = TimeoutFunction(agent.getAction, int(self.rules.getMoveTimeout(agentIndex)) - int(move_time))
                    try:
                        start_time = time.time()
                        if skip_action:
                            raise TimeoutFunctionException()
                        action = timed_func(observation)
                    except TimeoutFunctionException:
                        print("Agent %d timed out on a single move!" % agentIndex)
                        self.agentTimeout = True
                        self._agentCrash(agentIndex, quiet=True)
                        self.unmute()
                        return

                    move_time += time.time() - start_time

                    if move_time > self.rules.getMoveWarningTime(agentIndex):
                        self.totalAgentTimeWarnings[agentIndex] += 1
                        print("Agent %d took too long to make a move! This is warning %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex]))
                        if self.totalAgentTimeWarnings[agentIndex] > self.rules.getMaxTimeWarnings(agentIndex):
                            print("Agent %d exceeded the maximum number of warnings: %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex]))
                            self.agentTimeout = True
                            self._agentCrash(agentIndex, quiet=True)
                            self.unmute()

                    self.totalAgentTimes[agentIndex] += move_time

                    if self.totalAgentTimes[agentIndex] > self.rules.getMaxTotalTime(agentIndex):
                        print("Agent %d ran out of time! (time: %1.2f)" % (agentIndex, self.totalAgentTimes[agentIndex]))
                        self.agentTimeout = True
                        self._agentCrash(agentIndex, quiet=True)
                        self.unmute()
                        return

                    self.unmute()
                except Exception:
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return
            else:
                try:
                    action = agent.getAction(observation)
                except Exception:
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return

            self.unmute()

            # Execute action
            self.moveHistory.append((agentIndex, action))
            if self.catchExceptions:
                try:
                    self.state = self.state.generateSuccessor(agentIndex, action)
                except Exception:
                    self.mute(agentIndex)
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return
            else:
                self.state = self.state.generateSuccessor(agentIndex, action)

            # Update display
            try:
                self.display.update(self.state.data)
            except Exception:
                # If the display fails, print traceback and continue or abort depending on catchExceptions
                if not self.catchExceptions:
                    raise
                else:
                    traceback.print_exc()
                    self._agentCrash(agentIndex)
                    return

            # Let rules process (win/lose etc.)
            try:
                self.rules.process(self.state, self)
            except Exception:
                self._agentCrash(agentIndex)
                return

            # Track moves (some rules increment differently; keep simple)
            if agentIndex == numAgents - 1:
                self.numMoves += 1

            # Next agent
            agentIndex = (agentIndex + 1) % numAgents

            if _BOINC_ENABLED:
                try:
                    boinc.set_fraction_done(self.getProgress())
                except Exception:
                    pass

        # Inform learning agents of result
        for agentIndex, agent in enumerate(self.agents):
            if "final" in dir(agent):
                try:
                    self.mute(agentIndex)
                    agent.final(self.state)
                    self.unmute()
                except Exception:
                    if not self.catchExceptions:
                        raise
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return

        try:
            self.display.finish()
        except Exception:
            # ignore display finish exceptions
            pass

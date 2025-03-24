# baselineTeam.py
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

# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Actions
from game import Directions
from util import nearestPoint

#################
# Team creation #
#################
redTeam = False
blueTeam = False

def createTeam(firstIndex, secondIndex, isRed, 
               first='OffensiveHeuristicAgent', second='DefensiveHeuristicAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
    global redTeam, blueTeam
    first = 'OffensiveHeuristicAgent'
    second = 'DefensiveHeuristicAgent'
    if isRed:
        redTeam = True
    else:
        blueTeam = True
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########


class OffensiveHeuristicAgent(CaptureAgent):
    """
    A reflex agent that seeks food, and afterwards returns to base to score a point.
    """

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.foodAtStart = len(self.getFood(gameState).asList())

    def getFeatures(self, gameState, action, isPacman):
        """
        Offensive agent (Pacman) has two features:
        1. score of successor state represented by the number of food left after taking that action.
        The minus sign is used to prioritize states with fewer food pellets remaining
        2. The second feature is the distance to the closest food pellet with respect to the agent
        """
        features = util.Counter()
        successor = gameState.generateSuccessor(self.index, action)
        foodList = self.getFood(successor).asList()

        if isPacman:
            # Calculate minimum distance to food for Pacman
            currentPosition = successor.getAgentState(self.index).getPosition()
            if foodList:  # Ensure foodList is not empty
                minDist = min([self.getMazeDistance(currentPosition, food) for food in foodList])
                features['minDistToFood'] = minDist
                # Increment 'getFood' feature if Pacman eats food
                features['getFood'] = 1 if len(foodList) < len(self.getFood(gameState).asList()) else 0
        else:
            features['successorScore'] = -len(foodList)  # self.getScore(successor)

            # Compute distance to the nearest food
            if len(foodList) > 0:  # This should always be True,  but better safe than sorry
                myPos = successor.getAgentState(self.index).getPosition()
                minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
                features['distanceToFood'] = minDistance

        return features

    def pacmanRetreating(self, actions, gameState):
        bestDist = 9999
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            # Simulate the action and check if the resulting position is valid
            if pos2 is not None:
                # If the resulting position is valid, calculate the distance to the starting position
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist and action != 'Stop':
                    print("Action {}, distance {} < {}" .format(action, dist, bestDist))
                    bestAction = action
                    bestDist = dist
        return bestAction
    
    def searchFood(self, gameState, actions):
        bestScore = float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            # Compute the score of the successor state based on the A* algorithm
            # The score represents the distance (number of moves) required for the agent to reach the
            # closest food pellet
            score = self.astar_heuristic(successor)

            # 5. Select successor state with lower score
            if score < bestScore:
                bestScore = score
                bestAction = action
        return bestAction


    def isDeadEnd(self, gameState, depth, enemyGhosts, ghostSteps=3):
        """
        Checks if the current state is a dead end up to a certain depth, considering the movement of enemy ghosts.
        """
        if depth == 0:
            return True

        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')  # Ignore stopping action

        if len(actions) <= 1:
            return True  # Dead end reached
        
        # Check recursively for dead ends in successor states
        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            
            # Check if the ghost can reach the Pacman within a certain number of steps
            if any(self.isGhostNear(successor, ghost, ghostSteps) for ghost in enemyGhosts):
                continue  # Skip this action if the ghost can reach the Pacman
            
            if self.isDeadEnd(successor, depth - 1, enemyGhosts, ghostSteps):
                return True
        
        return False

    def isGhostNear(self, gameState, ghostIndex, steps):
        """
        Checks if the ghost is near the Pacman within a certain number of steps.
        """
        ghostPosition = gameState.getAgentPosition(ghostIndex)
        pacmanPosition = gameState.getAgentPosition(self.index)

        
        if ghostPosition is None or pacmanPosition is None:
            return False  # If positions are not available, consider ghost not near
        
        distance = self.getMazeDistance(ghostPosition, pacmanPosition)
        return distance <= steps


    def escapeFromGhost(self, gameState, actions, enemyGhosts):
        """
        Chooses the action that allows the Pacman agent to escape from the enemy ghosts while still searching for food.
        """
        bestScore = float('-inf')
        bestAction = None

        # Minimize distance to food and maximize distance to ghost
        bestDistanceToFood = float('inf')
        bestDistanceToGhost = float('-inf')

        for action in actions: 
            successor = self.getSuccessor(gameState, action)
            #score = 0

            # Calculate the distance to the nearest food pellet
            foodList = self.getFood(successor).asList()
            if len(foodList) > 0:
                myPos = successor.getAgentState(self.index).getPosition()
                minDistanceToFood = min([self.getMazeDistance(myPos, food) for food in foodList])
                #score -= minDistanceToFood

            # Calculate the distance to the nearest enemy ghost
            minDistanceToGhost = min([self.getMazeDistance(successor.getAgentPosition(self.index),
                                                        gameState.getAgentPosition(ghost))
                                    for ghost in enemyGhosts])

            
            isDeadEnd = self.isDeadEnd(successor, 3, enemyGhosts)  # Check for dead ends up to 3 steps ahead
            print("Action {}: minDistToFodd = {}, minDistToGhost = {}, dead end: {}" 
                  .format(action, minDistanceToFood, minDistanceToGhost, isDeadEnd))
            if minDistanceToFood <= bestDistanceToFood and minDistanceToGhost >= 2 and action != 'Stop':
                if minDistanceToGhost == 2 or minDistanceToGhost == 3:
                    if not isDeadEnd:
                            bestDistanceToFood = minDistanceToFood
                            bestAction = action
                else:
                    bestDistanceToFood = minDistanceToFood
                    bestAction = action
                

        print("Best action when escaping: ", bestAction)
        return bestAction


    def chooseAction(self, gameState):
        # This method uses the heuristic function to estimate the cost to reach the goal
        # Return the action that leads to the next best state based on the A* algorithm

        actions = gameState.getLegalActions(self.index)
        bestAction = None

        agentState = gameState.getAgentState(self.index)
        isPacman = agentState.isPacman
        if redTeam:
            foodEaten = self.foodAtStart - len(self.getFood(gameState).asList()) - gameState.getScore()
        else:
            foodEaten = self.foodAtStart - len(self.getFood(gameState).asList()) + gameState.getScore()
        print("Food eaten: ", foodEaten)

        currentPosition = gameState.getAgentState(self.index).getPosition()
        enemyGhosts = [ghost for ghost in self.detectEnemyGhost(gameState)
                        if self.getMazeDistance(currentPosition, gameState.getAgentPosition(ghost)) <= 4]
        if isPacman:
            # Case 1: Pacman Retreating because he has collected one food pellet
            if foodEaten % 1 == 0 and foodEaten != 0:
                print("Food collected: pacman retreating")
                bestAction = self.pacmanRetreating(actions, gameState)
            # Case 2: Pacman escaping from nearby ghost
            elif enemyGhosts:
                print("Pacman escaping from enemy")
                bestAction = self.escapeFromGhost(gameState, actions, enemyGhosts)
            # Case 3: Pacman is searching for food
            else:
                print("Pacman searching for food")
                bestAction = self.searchFood(gameState, actions)

        else:
            # Case 4: Ghost searching for food
            print("Ghost searching for food")
            bestAction = self.searchFood(gameState, actions)
        
        # If no best action was found with the A* algorithm we randomly choose between best actions
        if bestAction == 'Stop' or bestAction is None:
            agentState = gameState.getAgentState(self.index)
            isPacman = agentState.isPacman

            evaluations = [self.evaluate(gameState, a, isPacman) for a in actions]
            maxEvaluation = max(evaluations)
            bestActions = [action for action, value in zip(actions, evaluations) if value == maxEvaluation]
            bestAction = random.choice(bestActions)

        print("Best Action: ", bestAction)
        print("\n")
        #input("Enter")
        return bestAction

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def astar_heuristic(self, gameState):
        food = self.getFood(gameState).asList()
        currentPosition = gameState.getAgentState(self.index).getPosition()

        if not food:
            return 0  # No more food, no cost

        # Use A* search to find the closest food pellet
        queue = util.PriorityQueue()
        visited = set()

        for pellet in food:
            distance = self.getMazeDistance(currentPosition, pellet)
            queue.push((pellet, distance), distance)

        while not queue.isEmpty():
            position, cost = queue.pop()
            if position not in visited:
                visited.add(position)
                if position in food:
                    return cost
                for successor in self.getSuccessors(position):
                    newPosition, newCost = successor
                    if newPosition not in visited:
                        totalCost = cost + newCost
                        queue.push((newPosition, totalCost), totalCost)

        return float('inf')  # Couldn't find any food, return infinite cost


    def getSuccessors(self, position):
        """
        Get successor positions for a given position.
        """
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                successors.append(((nextx, nexty), 1))  # Cost of moving to successor is 1
        return successors

    def evaluate(self, gameState, action, isPacman):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action, isPacman)
        weights = self.getWeights(isPacman)
        return features * weights

    def getWeights(self, isPacman):
        if isPacman:
            return {'minDistToFood': -1, 'getFood': 100}
        else:
            return {'successorScore': 100, 'distanceToFood': -1}

    def detectEnemyGhost(self, gameState):
        enemyGhosts = []
        for enemy in self.getOpponents(gameState):
            if (not gameState.getAgentState(enemy).isPacman) and gameState.getAgentState(enemy).scaredTimer == 0:
                # The scaredTimer != 0 implies that the ghost is vulnerable, for it to be considered and enemy
                # it has to be equal to 0
                if gameState.getAgentPosition(enemy) is not None:
                    enemyGhosts.append(enemy)
        return enemyGhosts
    


#######################################################


class DefensiveHeuristicAgent(CaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free.
    """

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.border = self.determineBorder(gameState)
        self.target = None  # To keep track of the target enemy
        self.patrolIndex = 0  # Index for the current patrol target

    def isPositionOnOurSide(self, position):
        """
        Checks if the given position is on the agent's side.
        """
        midWidth = self.getFoodYouAreDefending(self.getCurrentObservation()).width // 2
        if self.red:
            return position[0] < midWidth
        else:
            return position[0] >= midWidth

    def determineBorder(self, gameState):
        """
        Determines the border of the agent's territory to patrol.
        """
        midWidth = gameState.data.layout.width // 2
        if self.red:
            midWidth -= 1
        else:
            midWidth += 1

        border = [(midWidth, y) for y in range(gameState.data.layout.height) if not gameState.hasWall(midWidth, y)]
        return border

    def chooseAction(self, gameState):
        """
        Picks the action to take in the current state.
        Either patrol the border or chase the target enemy if there is one on our side of the board.
        """
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]

        if invaders:
            distances = [self.getMazeDistance(gameState.getAgentPosition(self.index), a.getPosition()) for a in
                         invaders]
            self.target = invaders[distances.index(min(distances))].getPosition()
        else:
            self.target = None

        if self.target:
            return self.moveToTarget(gameState, self.target)
        else:
            return self.patrolBorder(gameState)

    def heuristic(self, position, goal):
        """
        Heuristic function for A* search
        """
        return util.manhattanDistance(position, goal)

    def aStarSearch(self, gameState, start, goal):
        """
        A star search algorithm for finding the shortest path from start to goal
        """
        frontier = util.PriorityQueue()
        frontier.push((gameState, start, []), 0)

        costSoFar = {start: 0}

        while not frontier.isEmpty():
            current_gameState, current_position, directions = frontier.pop()

            if current_position == goal:
                return directions

            for next_action in current_gameState.getLegalActions(self.index):
                successor_state = current_gameState.generateSuccessor(self.index, next_action)

                next_pos = successor_state.getAgentPosition(self.index)
                newCost = costSoFar[current_position] + 1

                if next_pos not in costSoFar:
                    costSoFar[next_pos] = newCost
                    priority = newCost + self.heuristic(next_pos, goal)

                    action_list = list(directions)
                    action_list += [next_action]

                    frontier.push((successor_state, next_pos, action_list), priority)

        return []

    def moveToTarget(self, gameState, target):
        """
        Moves to the target position, using A* search algorithm
        """
        start = gameState.getAgentPosition(self.index)
        directions = self.aStarSearch(gameState, start, target)

        if len(directions) == 0:
            return Directions.STOP
        else:
            return directions[0]

    def patrolBorder(self, gameState):
        """
        Patrols the border if there's no target, moving back and forth along the border positions.
        """
        if self.target is None or self.target not in self.border:
            self.target = self.border[self.patrolIndex]
            self.patrolIndex = (self.patrolIndex + 1) % len(self.border)

        return self.moveToTarget(gameState, self.target)


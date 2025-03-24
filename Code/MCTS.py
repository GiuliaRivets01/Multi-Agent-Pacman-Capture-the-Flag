# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import math


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveMCTSAgent', second='DefensiveMCTSAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class MCTSNode(object):
    def __init__(self, state, agent, action, parent, enemyPostion, border):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.tree_depth = parent.tree_depth + 1 if parent else 0
        self.wins = 0
        self.visits = 1
        self.agent = agent
        self.p = 1


        self.possibleActions = [action for action in state.getLegalActions(agent.index) if action != 'Stop']
        self.untriedActions = self.possibleActions[:]
        self.border = border
        self.enemyPosition = enemyPostion

    def addChild(self, child):
        self.children.append(child)

    def expand(self):
        """
        Expand the node by adding all possible successors (children) to the tree.
        """
        '''pseudocode slide 33 combined with 34'''
        if self.tree_depth >= 15: return self

        if self.untriedActions != []:
            action = random.choice(self.untriedActions)
            self.untriedActions.remove(action)

            next_state = self.state.generateSuccessor(self.agent.index, action)
            child_node = MCTSNode(next_state, self.agent, action, self, self.enemyPosition, self.border)
            self.addChild(child_node)
            return child_node

        if util.flipCoin(self.p):
            next_node = self.best_child()
        else:
            next_node = random.choice(self.children)

        return next_node.expand()

    def simulate_and_evaluate(self):  # this function should do the same thing as call_reward in github from class MCTSNode
        """
        Simulate a random playthrough from the gameState and return the result.
        """
        '''pseudocode slide 36'''
        current_position = self.state.getAgentPosition(self.agent.index)
        if current_position == self.state.getInitialAgentPosition(self.agent.index):
            # the agent is not making any progress
            return -1000

        # Calculate the minimum distance from the current position to the border positions
        min_distance_to_border = min(
            [self.agent.getMazeDistance(current_position, border_position) for border_position in self.border])

        # Since closer distances to the border are better, we use a negative weight for the distance.
        # Multiplying by -1 to convert smaller distances to larger rewards (or less negative).
        distance_weight = -1
        reward = min_distance_to_border * distance_weight

        return reward

    def backpropagate(self, reward):
        """
        Backpropagate the result of a simulation up the tree to the root.
        """
        '''pseudocode slide 37'''
        self.visits += 1
        self.wins += reward
        if self.parent is not None:
            self.parent.backpropagate(reward)

    '2 FUNCTIONS FOR SELECTING THE BEST CHILD'

    def best_child(self, c=1):  # this one considers exploration too
        """ Choose the best child"""

        '''pseudocode slide 35'''
        best_score = float('-inf')
        best_child = None
        for child in self.children:
            if child.visits == 0:
                continue
            exploit = child.wins / child.visits
            explore = math.sqrt(2.0 * math.log(self.visits) / child.visits)
            score = exploit + c*explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def best_child_only_exploitation(self):
        best_score = float('-inf')
        best_child = None
        for child in self.children:
            score = child.wins / child.visits
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def mctsChooseAction(self, iterations=1000):
        """
        Performs the MCTS algorithm, iterating a specified number of times.
        Returns the best action to take from the current gameState.
        """
        '''pseudocode slide 32'''

        # the number of iterations is the budget
        for _ in range(iterations):
            leaf = self.expand()  # select = tree policy
            simulation_reward = leaf.simulate_and_evaluate()  # simulate = defaultPolicy
            leaf.backpropagate(simulation_reward)

        best_child = self.best_child()
        return best_child.action


# ----------------------------------------------------------------------------------------------
##########
# Agents #
##########

class OffensiveMCTSAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.border = self.determineBorder(gameState)

    def determineBorder(self, gameState):
        midWidth = gameState.data.layout.width // 2

        if self.red:
            midWidth -= 1
        else:
            midWidth += 1

        # Filter out wall positions
        border = [(midWidth, y) for y in range(gameState.data.layout.height) if not gameState.hasWall(midWidth, y)]
        return border


    "Functions for ghosts detection"
    def detectEnemyNearBy(self, gameState):
        currentPosition = gameState.getAgentPosition(self.index)
        enemyGhosts = [ghost for ghost in self.detectEnemyGhost(gameState)
                       if self.getMazeDistance(currentPosition, gameState.getAgentPosition(ghost)) <= 5]
        return enemyGhosts

    def detectEnemyGhost(self, gameState):
        enemyGhosts = []
        for enemy in self.getOpponents(gameState):
            if (not gameState.getAgentState(enemy).isPacman) and gameState.getAgentState(enemy).scaredTimer == 0:
                # the scaredTimmer != 0 implies that the ghost is vulnerable, for it to be considered and enemy it has to be equal to 0
                if gameState.getAgentPosition(enemy) != None:
                    enemyGhosts.append(enemy)
        return enemyGhosts

    def chooseAction(self, gameState):
        border = self.determineBorder(gameState)

        actions = gameState.getLegalActions(self.index)
        agentState = gameState.getAgentState(self.index)

        numCarrying = agentState.numCarrying
        isPacman = agentState.isPacman

        food = self.getFood(gameState).asList()

        'Check for enemy ghosts closer than 3 steps'
        enemiesNearBy = [gameState.getAgentPosition(ghost) for ghost in self.detectEnemyNearBy(gameState)]

        # Use MCTS or evaluation based on the game state
        if len(food) < 2 or numCarrying > 7 or enemiesNearBy:
            rootNode = MCTSNode(gameState, self, None, None, enemiesNearBy, border)
            bestAction = rootNode.mctsChooseAction()
        else:
            # Evaluate actions using a unified function
            evaluations = [self.evaluate(gameState, a, isPacman, enemiesNearBy) for a in actions]
            maxEvaluation = max(evaluations)
            bestActions = [action for action, value in zip(actions, evaluations) if value == maxEvaluation]
            bestAction = random.choice(bestActions)

        return bestAction

    def evaluate(self, gameState, action, isPacman, approachingGhosts):
        """
        Unified evaluation function considering both offensive and defensive aspects.
        """
        features = self.getFeatures(gameState, action, isPacman, approachingGhosts)
        weights = self.getWeights(isPacman)
        return sum(features[f] * weights[f] for f in features)


    def getFeatures(self, gameState, action, isPacman, approachingGhosts):
        """
        Extracts relevant features from the gameState given an action.
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
            # Use 'distanceToFood' in a defensive context, for example, to stay close to food as a ghost
            currentPosition = successor.getAgentState(self.index).getPosition()
            if foodList:
                minDist = min([self.getMazeDistance(currentPosition, food) for food in foodList])
                features['distanceToFood'] = minDist
            features['successorScore'] = self.getScore(successor)

        return features

    def getWeights(self, isPacman):
        """
        Returns weights for the features based on whether the agent is Pacman or ghost.
        """
        if isPacman:
            return {'minDistToFood': -1, 'getFood': 100}
        else:
            return {'successorScore': 100, 'distanceToFood': -1}


class DefensiveMCTSAgent(CaptureAgent):
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
            Chooses actions aiming at defending the territory by intercepting intruders or patrolling the border.
            """
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]

            if invaders:
                distances = [self.getMazeDistance(gameState.getAgentPosition(self.index), a.getPosition()) for a in
                             invaders]
                self.target = invaders[distances.index(min(distances))].getPosition()
            else:
                # If there's no invader, continue or start patrolling
                self.target = None  # Reset target to ensure patrol behavior updates

            # If there's a target, move towards it, otherwise patrol the border
            if self.target:
                return self.moveToTarget(gameState, self.target)
            else:
                return self.patrolBorder(gameState)

        def moveToTarget(self, gameState, target):
            """
            Moves towards the target position, ensuring the agent does not cross into enemy territory.
            """
            actions = gameState.getLegalActions(self.index)
            bestAction = None
            shortestDist = float('inf')

            for action in actions:
                successor = gameState.generateSuccessor(self.index, action)
                pos2 = successor.getAgentPosition(self.index)
                # Check if the next position is on our side before considering the action
                if self.isPositionOnOurSide(pos2):
                    dist = self.getMazeDistance(pos2, target)
                    if dist < shortestDist:
                        bestAction = action
                        shortestDist = dist

            return bestAction



        def patrolBorder(self, gameState):
            """
            Patrols the border if there's no target, moving back and forth along the border positions.
            """
            if self.target is None or self.target not in self.border:
                self.target = self.border[self.patrolIndex]
                self.patrolIndex = (self.patrolIndex + 1) % len(self.border)

            return self.moveToTarget(gameState, self.target)
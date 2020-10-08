# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for i in range(0, self.iterations):
            k_counter = util.Counter()
            for state in self.mdp.getStates():
                temp_counter = util.Counter()
                for action in self.mdp.getPossibleActions(state):
                    temp_counter[action] = self.computeQValueFromValues(state, action)
                max_action = temp_counter.argMax()
                k_counter[state] = temp_counter[max_action]
            self.values = k_counter

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        sum = 0
        for tuple in self.mdp.getTransitionStatesAndProbs(state, action):
            nextState, T = tuple
            R = self.mdp.getReward(state, action, nextState)
            sum += T * (R + self.discount * self.values[nextState])
        return sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        temp_counter = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            num = 0
            for tuple in self.mdp.getTransitionStatesAndProbs(state, action):
                nextState, T = tuple
                R = self.mdp.getReward(state, action, nextState)
                num += T * (R + self.discount*self.values[nextState])
            temp_counter[action] = num
        return temp_counter.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        n = len(states)
        for i in range(0, self.iterations):
            k_counter = util.Counter()
            state = states[i % n]
            if self.mdp.isTerminal(state):
                continue
            temp_counter = util.Counter()
            for action in self.mdp.getPossibleActions(state):
                num = 0
                for tuple in self.mdp.getTransitionStatesAndProbs(state, action):
                    nextState, T = tuple
                    R = self.mdp.getReward(state, action, nextState)
                    num += T * (R + self.discount*self.values[nextState])
                temp_counter[action] = num
            max_action = temp_counter.argMax()
            self.values[state] = temp_counter[max_action]

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        self.qValues = util.Counter()
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def getPredecessors(self):
        dict = {}
        states = self.mdp.getStates()
        for state in states:
            dict[state] = set([])

        for state in states:
            for action in self.mdp.getPossibleActions(state):
                for nextState, T in self.mdp.getTransitionStatesAndProbs(state, action):
                    if T > 0:
                        dict[nextState].add(state)
        return dict

    def runValueIteration(self):
        predecessors = self.getPredecessors()
        pq = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                temp = util.Counter()
                for action in self.mdp.getPossibleActions(state):
                    num = 0
                    for nextState, T in self.mdp.getTransitionStatesAndProbs(state, action):
                        R = self.mdp.getReward(state, action, nextState)
                        num += T * (R + self.discount * self.values[nextState])
                    temp[action] = num
                maxAction = temp.argMax()
                maxValue = temp[maxAction]
                diff = abs(self.getValue(state) - maxValue)
                pq.push(state, -diff)

        for i in range(0, self.iterations):
            if pq.isEmpty(): break
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                temp = util.Counter()
                for action in self.mdp.getPossibleActions(state):
                    temp[action] = self.computeQValueFromValues(state, action)
                max_action = temp.argMax()
                self.values[state] = temp[max_action]
            for p in predecessors[state]:
                temp_counter = util.Counter()
                for action in self.mdp.getPossibleActions(p):
                    num = 0
                    for nextState, T in self.mdp.getTransitionStatesAndProbs(p, action):
                        R = self.mdp.getReward(p, action, nextState)
                        num += T * (R + self.discount*self.values[nextState])
                    temp_counter[action] = num
                max_action = temp_counter.argMax()
                max_val = temp_counter[max_action]
                diff = abs(self.values[p] - max_val)
                if diff > self.theta:
                    pq.update(p, -diff)

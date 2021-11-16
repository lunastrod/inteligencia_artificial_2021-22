# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game


pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()

def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** YOUR CODE HERE ***"
    A=logic.Expr('A')
    B=logic.Expr('B')
    C=logic.Expr('C')
    expr1=A|B
    expr2=~A % (~B | C)
    expr3=logic.disjoin(~A, ~B, C)
    return logic.conjoin(expr1,expr2,expr3)

def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"
    A=logic.Expr('A')
    B=logic.Expr('B')
    C=logic.Expr('C')
    D=logic.Expr('D')
    """
    C⇔B∨D
    A⇒¬B∧¬D
    ¬ (B∧¬C) ⇒A
    ¬D⇒C
    
    """
    expr1=C%(B|D)
    expr2=A>>(~B&~D)
    expr3=~(B&~C)>>A
    expr4=~D>>C
    return logic.conjoin(expr1,expr2,expr3,expr4)

def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"



    WA1 = logic.PropSymbolExpr("WumpusAlive[1]")
    WA0 = logic.PropSymbolExpr("WumpusAlive[0]")
    WB0 = logic.PropSymbolExpr("WumpusBorn[0]")
    WK0 = logic.PropSymbolExpr("WumpusKilled[0]")

    expr1=WA1%((WA0&~WK0) | (~WA0&WB0))
    expr2=~(WA0 & WB0)
    expr3=WB0
    return logic.conjoin(expr1,expr2,expr3)

def modelToString(model):
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to logic.pycoSAT.
    """
    if model == False:
        return "False" 
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)

def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"
    #print("findmodel")
    #print(sentence)
    cnf = logic.to_cnf(sentence)
    #print(cnf)
    sat = logic.pycoSAT(cnf)
    #print(sat)
    return sat

def atLeastOne(literals):
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single 
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    return logic.disjoin(literals)

def atMostOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    #hecho con https://www.charlie-coleman.com/experiments/kmap/
    conjunctions = []
    for i in range(len(literals)):
        for j in range(0,i):
            conjunctions.append(logic.disjoin(~literals[i],~literals[j]))

    #print(conjunctions)
    return logic.conjoin(conjunctions)



def exactlyOne(literals):
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    #hecho con https://www.charlie-coleman.com/experiments/kmap/
    conjunctions = []
    conjunctions.append(logic.disjoin(literals))
    for i in range(len(literals)):
        for j in range(0,i):
            conjunctions.append(logic.disjoin(~literals[i],~literals[j]))

    #print(conjunctions)
    return logic.conjoin(conjunctions)



def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"

    result = []
    #print("model:\n",model,"\n")
    for i in model.keys():
        if model[i]:
            a = logic.parseExpr(i)
            if a[0] in actions:
                result.append(a)

    #print("extract actions:\n",result,"\n")

    result = sorted(result, key=lambda mod: int(mod[1]))
    #print("sort:\n",result,"\n")
    
    for i in range(len(result)):
        result[i]=result[i][0]
    #print("list of actions:\n",result,"\n")
    return result

def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    "*** YOUR CODE HERE ***"
    current = logic.PropSymbolExpr(pacman_str, x, y, t)

    neighbors = []

    if(walls_grid[x-1][y] == False):
        prev_position = logic.PropSymbolExpr(pacman_str, x-1, y, t-1)
        action = logic.PropSymbolExpr('East', t-1)
        state = logic.conjoin(prev_position, action)
        neighbors.append(state)

    if(walls_grid[x+1][y] == False):
        prev_position = logic.PropSymbolExpr(pacman_str, x+1, y, t-1)
        action = logic.PropSymbolExpr('West', t-1)
        state = logic.conjoin(prev_position, action)
        neighbors.append(state)

    if(walls_grid[x][y-1] == False):
        prev_position = logic.PropSymbolExpr(pacman_str, x, y-1, t-1)
        action = logic.PropSymbolExpr('North', t-1)
        state = logic.conjoin(prev_position, action)
        neighbors.append(state)

    if(walls_grid[x][y+1] == False):
        prev_position = logic.PropSymbolExpr(pacman_str, x, y+1, t-1)
        action = logic.PropSymbolExpr('South', t-1)
        state = logic.conjoin(prev_position, action)
        neighbors.append(state)


    #print("\n\n\nstart")
    #print("neighbors:\n",neighbors,"\n")
    prev_states = atLeastOne(neighbors)
    #print("prev_states:\n",prev_states,"\n")

    final_axiom = current % prev_states
    #print("final axiom:\n",final_axiom,"\n")
    return final_axiom


def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    x0, y0 = problem.getStartState()
    xg, yg = problem.getGoalState()
    
    "*** YOUR CODE HERE ***"
    """
    walls = problem.walls
    
    MAX_TIME_STEP = 50
    actions = ['North', 'East', 'South', 'West']
    width, height = problem.getWidth(), problem.getHeight()
    initial_state = problem.getStartState()
    goal_state = problem.getGoalState()
    expression = list()


    # CAN ONLY START AT ONE LOCATION
    initial_positions=[]
    for x in range(1, width+1) :
        for y in range(1, height+1) :
            initial_positions.append(logic.PropSymbolExpr("P", x, y, 0))
            if (x, y) == initial_state:
                if expression:
                    v = expression.pop()
                    expression.append(logic.conjoin(v,logic.PropSymbolExpr("P", x, y, 0)))
                else:
                    expression.append(logic.Expr(logic.PropSymbolExpr("P", x, y, 0)))
            else:
                if expression:
                    v = expression.pop()
                    expression.append(logic.conjoin(v,logic.Expr("~", logic.PropSymbolExpr("P", x, y, 0))))
                else:
                    expression.append(logic.Expr("~", logic.PropSymbolExpr("P", x, y, 0)))
    initial = expression[0]
    print("\ninitial:\n",initial)
    initial=exactlyOne(initial_positions)
    print("\ninitial:\n",initial)


    successors = []
    exclusion = []
    for t in range(MAX_TIME_STEP):
        succ = []
        ex = []
        suc = []
        if t > 0:
            for x in range(1, width + 1):
                for y in range(1, height + 1):
                    if (x, y) not in walls.asList():
                        succ += [pacmanSuccessorStateAxioms(x, y, t, walls)]
            suc = logic.conjoin(succ) #or every place at t 
            if successors:
                success = logic.conjoin(suc, logic.conjoin(successors)) #combine with previous successors
            else:
                success = suc


            # ONLY ONE ACTION CAN BE TAKEN
            for action in actions: #exclusion axioms
                ex.append(logic.PropSymbolExpr(action, t-1))
            n = exactlyOne(ex)
            exclusion.append(n)
            exclus = logic.conjoin(exclusion)


            # GOAL TEST
            goal = logic.conjoin(logic.PropSymbolExpr("P", goal_state[0], goal_state[1], t+1), pacmanSuccessorStateAxioms(goal_state[0], goal_state[1], t+1, walls))

            
            # CONJOIN AND FIND MODEL
            j = findModel(logic.conjoin(initial, goal, exclus, success)) #and them together

        else:
            goal = logic.conjoin(logic.PropSymbolExpr("P", goal_state[0], goal_state[1], t+1), pacmanSuccessorStateAxioms(goal_state[0], goal_state[1], t+1, walls))
            j = findModel(logic.conjoin(initial, goal))
        if j is not False:
            return extractActionSequence(j, actions)
        if suc:
            successors.append(suc)
    return None


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    (x0, y0), food_locations = problem.getStartState()
    food_list = food_locations.asList()    
    walls_list = walls.asList()


    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan
#fglp = foodGhostLogicPlan

# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
    

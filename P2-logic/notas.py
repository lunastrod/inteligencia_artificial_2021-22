def sentence4():
    """
    Define el método sentence4().
    Usa  el constructor logic.PropSymbolExpr para crear los símbolos 
        Agente[1,1,0], Agente[2,1,1], Agente[Norte,0] y Agente[Este,0].
    Crea una instancia de logic.Expr que codifique las siguientes tres oraciones en lógica proposicional en
    este orden y sin ninguna simplificación:
        1. El Agente está en la posición (2,1) en el tiempo 1 si y solo si estaba en la posición (1,1) en
            el tiempo 0 y se movió hacia el Este en el tiempo 0 y no se movió hacia el Norte en el
            tiempo 0.
        2. El Agente está en la posición (2,1) en el tiempo 1.
        3. El Agente en el tiempo 0 solo se ha podido mover al Este o al Norte.
    Devuelve la conjunción de las sentencias anteriores
    """

    A110 = logic.PropSymbolExpr("Agente[1,1,0]")
    A211 = logic.PropSymbolExpr("Agente[2,1,1]")
    ANo0 = logic.PropSymbolExpr("Agente[Norte,0]")
    AEs0 = logic.PropSymbolExpr("Agente[Este,0]")

    expr1=(A211)%logic.conjoin(A110, AEs0, ~ANo0)
    expr2=A211
    expr3=(ANo0 & ~AEs0) | (~ANo0 & AEs0) #tambien podría usar exactlyOne() que ahí está en cnf
    return logic.conjoin(expr1,expr2,expr3)

def allTrue(literales):
    #todos true es una puerta and
    return logic.conjoin(literales)

def noOne(literales):
    #todos false es una puerta and con todas las entradas negadas, o una puerta nor.
    return ~logic.disjoin(literales)















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
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    aaa = logic.disjoin(A,B)
    a_or_b = ~A % (~B | C)
    not_a = logic.disjoin(~A,~B,C)
    return logic.conjoin(aaa,a_or_b,not_a)




def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    D = logic.Expr('D')
    aaa = C % (B | D)
    a_or_b = A >> (~B & ~D)
    not_a = ~(B & ~C) >> A
    not_d = ~D >> C
    return logic.conjoin(aaa,a_or_b,not_a,not_d)


def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):
    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.
    The Wumpus cannot both be alive at time 0 and be born at time 0.
    The Wumpus is born at time 0.
    """
    a = logic.PropSymbolExpr("WumpusAlive[1]")
    b = logic.PropSymbolExpr("WumpusAlive[0]")
    c = logic.PropSymbolExpr("WumpusBorn[0]")
    d = logic.PropSymbolExpr("WumpusKilled[0]")
    alive = a % ((b & ~d) | (~b & c))
    cant = ~(b & c)
    born = c
    return logic.conjoin(alive,cant,born) 



def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    # print sentence
    # print "his"
    a = logic.to_cnf(sentence)
    # print "pis"
    b = logic.pycoSAT(a)
    # print "dis"
    if str(b) == "FALSE":
        # print "wis"
        return False
    # print "jis"
    return b

def atLeastOne(literals) :
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
    return logic.disjoin(literals)


def atMostOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    conjunctions = []
    for literal in literals:
        not_literal = ~literal

        # Disjoin literal with NOT(literal) for every other element besides this literal
        # and add it to the list to be conjoined
        for inner_literal in literals:
            if literal != inner_literal:
                not_inner_literal = ~inner_literal
                disjunction = logic.disjoin(not_literal, not_inner_literal)
                conjunctions.append(disjunction)

    return logic.conjoin(conjunctions)


def exactlyOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    # print literals
    conjunctions = []
    one_must_be_true_list = []
    for literal in literals:
        not_literal = ~literal
        one_must_be_true_list.append(literal)

        # Disjoin literal with NOT(literal) for every other element besides this literal
        # and add it to the list to be conjoined
        reached_literal = False
        for inner_literal in literals:
            if (reached_literal):
                not_inner_literal = ~inner_literal
                disjunction = logic.disjoin(not_literal, not_inner_literal)
                conjunctions.append(disjunction)
            if literal == inner_literal:
                reached_literal = True

    # Add the expression that states at least one of the literals must be true
    one_must_be_true = logic.disjoin(one_must_be_true_list)
    conjunctions.append(one_must_be_true)

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
    >>> print plan
    ['West', 'South', 'North']
    """
    models = []
    final = []
    # print "hur"
    for i in model.keys():
        # print "wo"
        if model[i]:
            a = logic.PropSymbolExpr.parseExpr(i)
            if a[0] in actions:
                models.append(a)
    p = sorted(models, key=lambda mod: int(mod[1]))
    # print "hi"
    for m in p:
        final.append(m[0])

    # print "returning"
    return final

def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    current = logic.PropSymbolExpr(pacman_str, x, y, t)

    neighbors = []

    if walls_grid[x-1][y] == False:
        prev_position = logic.PropSymbolExpr(pacman_str, x-1, y, t-1)
        action = logic.PropSymbolExpr('East', t-1)
        state = logic.conjoin(prev_position, action)
        neighbors.append(state)

    if walls_grid[x+1][y] == False:
        prev_position = logic.PropSymbolExpr(pacman_str, x+1, y, t-1)
        action = logic.PropSymbolExpr('West', t-1)
        state = logic.conjoin(prev_position, action)
        neighbors.append(state)

    if walls_grid[x][y-1] == False:
        prev_position = logic.PropSymbolExpr(pacman_str, x, y-1, t-1)
        action = logic.PropSymbolExpr('North', t-1)
        state = logic.conjoin(prev_position, action)
        neighbors.append(state)

    if walls_grid[x][y+1] == False:
        prev_position = logic.PropSymbolExpr(pacman_str, x, y+1, t-1)
        action = logic.PropSymbolExpr('South', t-1)
        state = logic.conjoin(prev_position, action)
        neighbors.append(state)

    prev_states = atLeastOne(neighbors)
    final_axiom = current % prev_states
    # print final_axiom
    return final_axiom

def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls

    MAX_TIME_STEP = 50
    actions = ['North', 'East', 'South', 'West']
    width, height = problem.getWidth(), problem.getHeight()
    initial_state = problem.getStartState()
    goal_state = problem.getGoalState()
    expression = list()


    # CAN ONLY START AT ONE LOCATION
    for x in range(1, width+1) :
        for y in range(1, height+1) :
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
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    MAX_TIME_STEP = 50
    actions = ['North', 'East', 'South', 'West']
    

    initial_state = problem.getStartState()
    # Pacman's initial location
    pacman_initial_location = initial_state[0]
    # Food locations
    food_locations = initial_state[1].asList()

    expression = list()

    for x in range(1, width + 1) :
        for y in range(1, height + 1) :
            if (x, y) == pacman_initial_location:
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
            for action in actions: #exclusion axioms
                ex.append(logic.PropSymbolExpr(action, t-1))
            n = exactlyOne(ex)
            exclusion.append(n)
            exclus = logic.conjoin(exclusion)
            food_locations_eaten = list()
            for food_particle in food_locations:
                food_particles = list()
                for i in range(0, t+1):
                    food_particles.append(logic.PropSymbolExpr("P", food_particle[0], food_particle[1], i))
                food_particles = logic.disjoin(food_particles)
                food_locations_eaten.append(food_particles)
            food_locations_eaten = logic.conjoin(food_locations_eaten)
            j = findModel(logic.conjoin(initial, food_locations_eaten, exclus, success)) #and them together
        else:
            food_locations_eaten = list()
            for food_particle in food_locations:
                food_locations_eaten.append(logic.PropSymbolExpr("P", food_particle[0], food_particle[1], 0))
            food_locations_eaten = logic.conjoin(food_locations_eaten)
            j = findModel(logic.conjoin(initial, food_locations_eaten))
        if j is not False:
            return extractActionSequence(j, actions)
        if suc:
            successors.append(suc)
    return None

def ghostPositionSuccessorStateAxioms(x, y, t, ghost_num, walls_grid):
    """
    Successor state axiom for patrolling ghost state (x,y,t) (from t-1).
    Current <==> (causes to stay) | (causes of current)
    GE is going east, ~GE is going west 
    """
    pos_str = ghost_pos_str+str(ghost_num)
    east_str = ghost_east_str+str(ghost_num)
    current = logic.PropSymbolExpr(pos_str, x, y, t)

    neighbors = []

    if walls_grid[x-1][y] == False:
        prev_position = logic.PropSymbolExpr(pos_str, x-1, y, t-1)
        action = logic.PropSymbolExpr(east_str, t-1)
        state = logic.conjoin(prev_position, action)
        neighbors.append(state)

    if walls_grid[x+1][y] == False:
        prev_position = logic.PropSymbolExpr(pos_str, x+1, y, t-1)
        action = ~logic.PropSymbolExpr(east_str, t-1)
        state = logic.conjoin(prev_position, action)
        neighbors.append(state)

    prev_states = atLeastOne(neighbors)
    if str(prev_states) == "FALSE":
        final_axiom = current % logic.PropSymbolExpr(pos_str, x, y, t-1)
    else:
        final_axiom = current % prev_states
    return final_axiom

def ghostDirectionSuccessorStateAxioms(t, ghost_num, blocked_west_positions, blocked_east_positions):
    """
    Successor state axiom for patrolling ghost direction state (t) (from t-1).
    west or east walls.
    Current <==> (causes to stay) | (causes of current)
    """
    pos_str = ghost_pos_str+str(ghost_num)
    east_str = ghost_east_str+str(ghost_num)
    west_positions = list()
    east_positions = list()
    blocked_west_positions_copy = blocked_west_positions[:]
    blocked_east_positions_copy = blocked_east_positions[:]

    while blocked_west_positions_copy:
        position = blocked_west_positions_copy.pop()
        west_positions.append(logic.PropSymbolExpr(pos_str, position[0], position[1], t))
   
    while blocked_east_positions_copy:
        position = blocked_east_positions_copy.pop()
        east_positions.append(logic.PropSymbolExpr(pos_str, position[0], position[1], t))
    
    west_positions = logic.disjoin(west_positions)
    east_positions = logic.disjoin(east_positions)

    conditions = (west_positions & ~logic.PropSymbolExpr(east_str, t-1)) | (~east_positions & logic.PropSymbolExpr(east_str, t-1))
    final_axiom = logic.PropSymbolExpr(east_str, t) % conditions

    return final_axiom 


def pacmanAliveSuccessorStateAxioms(x, y, t, num_ghosts):
    """
    Successor state axiom for patrolling ghost state (x,y,t) (from t-1).
    Current <==> (causes to stay) | (causes of current)
    """
    ghost_strs = [ghost_pos_str+str(ghost_num) for ghost_num in xrange(num_ghosts)]
    current = logic.PropSymbolExpr(pacman_str, x, y, t)
    ghosts = ghost_strs[:]
    neighbors = []

    k = []
    l = []
    while num_ghosts != 0:
        k += [logic.PropSymbolExpr(ghost_strs[num_ghosts-1], x, y, t-1) | logic.PropSymbolExpr(ghost_strs[num_ghosts-1], x, y, t)]
        num_ghosts -= 1
    m = ~logic.PropSymbolExpr(pacman_alive_str, t-1)

    prev_states = logic.disjoin(k)
    prev_states = logic.conjoin(logic.PropSymbolExpr(pacman_str, x, y, t), prev_states) | m
    final_axiom = ~logic.PropSymbolExpr(pacman_alive_str, t) % prev_states
    return final_axiom

def foodGhostLogicPlan(problem):
    """
    Given an instance of a FoodGhostPlanningProblem, return a list of actions that help Pacman
    eat all of the food and avoid patrolling ghosts.
    Ghosts only move east and west. They always start by moving East, unless they start next to
    and eastern wall. 
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    ghosts = problem.getGhostStartStates()
    ghost_positions = []
    ghost_rows = []
    for ghost in ghosts:
        p = ghost.getPosition()
        ghost_positions.append(p)
        ghost_rows.append(p[1])
    ghost_num = len(ghost_positions)
    MAX_TIME_STEP = 50
    actions = ['North', 'East', 'South', 'West']
    

    initial_state = problem.getStartState()
    # Pacman's initial location
    pacman_initial_location = initial_state[0]
    # Food locations
    food_locations = initial_state[1].asList()

    
    # GET THE BLOCKED POSITIONS TO PASS INTO GHOST DIRECTION SSA
    blocked_east_positions = []
    blocked_west_positions = []
    wall = walls.asList()
    # ble = problem.walls.asList()
    
    for x in range(0, width + 2):
        for y in range(0, height + 1):
            if (x, y) in wall:
                if (x+1, y) not in wall:
                    if x <= width:
                        blocked_west_positions.append((x+1, y))
                if (x-1, y) not in wall:
                    if x > 0:
                        blocked_east_positions.append((x-1, y))
  
    i = 0
    ghost_init = []
    ghost1pos = []
    ghost2pos = []
    expression = list()
    for x in range(1, width + 1) :
        for y in range(1, height + 1) :
            if (x, y) == pacman_initial_location:
                e = 0
                while e != ghost_num:
                    if ghost2pos:
                        ghost2pos = ghost2pos & ~logic.PropSymbolExpr(ghost_pos_str+str(e), x, y, 0)
                    else:
                        ~logic.PropSymbolExpr(ghost_pos_str+str(e), x, y, 0)
                    e += 1
                if expression:
                    # v = expression.pop()
                    expression = expression & logic.PropSymbolExpr("P", x, y, 0)
                else:
                    expression = logic.PropSymbolExpr("P", x, y, 0)
            if (x, y) in ghost_positions:
                east_str = ghost_east_str+str(i)
                j = 0
                while j != ghost_num:
                    if j != i:
                        if ghost2pos:
                            ghost2pos = ghost2pos & ~logic.PropSymbolExpr(ghost_pos_str+str(j), x, y, 0)
                        else:
                            ghost2pos = ~logic.PropSymbolExpr(ghost_pos_str+str(j), x, y, 0)
                    j += 1
                if (x, y) in blocked_east_positions:
                    if ghost_init:
                        # u = ghost_init.pop()
                        # r = ghost1pos.pop()
                        ghost_init = ghost_init & ~logic.PropSymbolExpr(east_str, 0)
                        ghost1pos = ghost1pos & logic.PropSymbolExpr(ghost_pos_str+str(i), x, y, 0)
                        i += 1
                    else:
                        ghost_init = ~logic.PropSymbolExpr(east_str, 0)
                        ghost1pos = logic.PropSymbolExpr(ghost_pos_str+str(i), x, y, 0)
                        i += 1
                else:
                    if ghost_init:
                        # u = ghost_init.pop()
                        # r = ghost1pos.pop()
                        ghost_init = ghost_init & logic.PropSymbolExpr(east_str, 0)
                        ghost1pos = ghost1pos & logic.PropSymbolExpr(ghost_pos_str+str(i), x, y, 0)
                        i += 1
                    else:
                        ghost_init = logic.PropSymbolExpr(east_str, 0)
                        ghost1pos = logic.PropSymbolExpr(ghost_pos_str+str(i), x, y, 0)
                        i += 1
            if (x, y) != pacman_initial_location:
                if (x, y) not in ghost_positions:
                    e = 0
                    while e != ghost_num:
                        if ghost2pos:
                            ghost2pos = ghost2pos & ~logic.PropSymbolExpr(ghost_pos_str+str(e), x, y, 0)
                        else:
                            ghost2pos = ~logic.PropSymbolExpr(ghost_pos_str+str(e), x, y, 0)
                        e += 1
                if expression:
                    # v = expression.pop()
                    expression = expression & logic.Expr("~", logic.PropSymbolExpr("P", x, y, 0))
                else:
                    expression = logic.Expr("~", logic.PropSymbolExpr("P", x, y, 0))
    
    # not_ghost = logic.conjoin(ghost2pos)

    initial = expression & ghost_init & ghost1pos & ghost2pos
    
    pacman_ssa = []
    pacman_alive_ssa = []
    ghost_position_ssa = []
    ghost_direction_ssa = []

    only_one_action = []
    
    for t in range(MAX_TIME_STEP):
        
        if t > 0:
            once = True
           
            for x in range(1, width + 1):
                for y in range(1, height + 1):
                    if not walls[x][y]:
                        pacman_ssa += [pacmanSuccessorStateAxioms(x, y, t, walls)]
                        if y in ghost_rows:
                            pacman_alive_ssa += [pacmanAliveSuccessorStateAxioms(x, y, t, ghost_num)]
                        
                        i = 0
                        while i != ghost_num:
                            if once:
                                ghost_direction_ssa += [ghostDirectionSuccessorStateAxioms(t, i, blocked_west_positions, blocked_east_positions)]
                            ghost_position_ssa += [ghostPositionSuccessorStateAxioms(x, y, t, i, walls)]
                            i += 1
                        once = False
            
            # ADD GHOST_DIRECTION_SSA
            i = 0
           
       
            # CONJOIN PACMAN_SSA
            pacman_ssa_conjoined = logic.conjoin(pacman_ssa)
            # CONJOIN PACMAN_ALIVE_SSA
            pacman_alive_ssa_conjoined = logic.conjoin(pacman_alive_ssa)
            # CONJOIN GHOST_POSITION_SSA
            ghost_position_ssa_conjoined = logic.conjoin(ghost_position_ssa)
            # CONJOIN GHOST_DIRECTION_SSA
            ghost_direction_ssa_conjoined = logic.conjoin(ghost_direction_ssa)

            
            # MAKES SURE ONLY ONE ACTION IS TAKEN 
            possible_actions = []
            one_action = []
            for action in actions: #exclusion axioms
                possible_actions.append(logic.PropSymbolExpr(action, t-1))
            one_action = exactlyOne(possible_actions)
            only_one_action.append(one_action)
            only_one_action_conjoined = logic.conjoin(only_one_action)

        
            # FIND OUT IF ALL THE FOOD HAS BEEN EATEN AS A GOAL TEST
            food_locations_eaten = list()
            for food_particle in food_locations:
                food_particles = list()
                for i in range(0, t+1):
                    food_particles.append(logic.PropSymbolExpr("P", food_particle[0], food_particle[1], i))
                food_particles = logic.disjoin(food_particles)
                food_locations_eaten.append(food_particles)
            food_locations_eaten = logic.conjoin(food_locations_eaten)
           
            # PACMAN IS ALIVE AT TIME T
        
            pacman_alive = logic.to_cnf(logic.PropSymbolExpr(pacman_alive_str, t))
            initial = logic.to_cnf(initial)
            food_locations_eaten = logic.to_cnf(food_locations_eaten)
            only_one_action_conjoined = logic.to_cnf(only_one_action_conjoined)
            pacman_ssa_conjoined = logic.to_cnf(pacman_ssa_conjoined)
            pacman_alive_ssa_conjoined = logic.to_cnf(pacman_alive_ssa_conjoined)
            ghost_position_ssa_conjoined = logic.to_cnf(ghost_position_ssa_conjoined)
            ghost_direction_ssa_conjoined = logic.to_cnf(ghost_direction_ssa_conjoined)
  
            j = logic.pycoSAT(pacman_alive & initial & food_locations_eaten & only_one_action_conjoined & pacman_ssa_conjoined & pacman_alive_ssa_conjoined 
                            & ghost_position_ssa_conjoined & ghost_direction_ssa_conjoined)
    
        else:
            food_locations_eaten = list()
            for food_particle in food_locations:
                food_locations_eaten.append(logic.PropSymbolExpr("P", food_particle[0], food_particle[1], 0))
            food_locations_eaten = logic.conjoin(food_locations_eaten)
            j = logic.pycoSAT(logic.conjoin(initial, food_locations_eaten))
        if j is not False:
            # for key, val in j.items():
            #     if val:
            #         print key
            return extractActionSequence(j, actions)
    return None

# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan
fglp = foodGhostLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)

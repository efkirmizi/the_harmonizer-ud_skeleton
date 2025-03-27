# Name Surname: Enis Furkan KIRMIZI
# Student ID: 150230740

import time
import numpy as np
from agent.agent import *

########################################################
######## YOUR CAN ADD ADDITIONAL FUNCTIONS HERE ########
########################################################

def matrix_to_tuple(matrix):
    """
        Transforms list[list] format to set(set())

        Args:
            list of lists
        Returns:
            set of sets
    """
    return tuple(tuple(row) for row in matrix)

def check_player_move_off_grid(player, x, y, matrix, p1_pos, p2_pos):
    """
        Checks if the player character move off the grid.
    """
    if player == 1:
        ppos = p1_pos
    else:
        ppos = p2_pos

    return 0 > ppos[0] + y or ppos[0] + y >= len(matrix[0]) \
        or 0 > ppos[1] + x or ppos[1] + x >= len(matrix)

def check_moves(x, y, matrix, p1_pos, p2_pos):
    """
        Checks if the moves are valid.
    """
    # If at least one of the players tries to move in a valid space, returns True.
    return not check_player_move_off_grid(1, x, y, matrix, p1_pos, p2_pos) \
        and matrix[p1_pos[0] + y][p1_pos[1] + x] != 4 \
        or not check_player_move_off_grid(2, -x, -y, matrix, p1_pos, p2_pos) \
        and matrix[p2_pos[0] - y][p2_pos[1] - x] != 4

def get_player_direction(player, x, y, matrix, p1_pos, p2_pos):
    """
        Gets the player character's direction.
    """
    ppos = p1_pos if player == 1 else p2_pos
    # The player character's initial direction. [x, y] for 1, [-x, -y] for 2.
    pdir = [x * (-1) ** (player + 1), y * (-1) ** (player + 1)]

    # If the player character tries to move off the grid or to a wall, it returns [0, 0] (no movement).
    if check_player_move_off_grid(player, pdir[0], pdir[1], matrix, p1_pos, p2_pos)\
            or matrix[ppos[0] + pdir[1]][ppos[1] + pdir[0]] == 4:
        return [0, 0]

    # If the player character tries to move to a conveyor, it moves the player character to the end of the conveyor.
    while matrix[ppos[0] + pdir[1]][ppos[1] + pdir[0]] in [5, 6, 7, 8]:
        conveyor_id = matrix[ppos[0] + pdir[1]][ppos[1] + pdir[0]] - 5
        # Determines the direction of the conveyor from its ID.
        conveyor_dir = [conveyor_id // 2 * (1 - 2 * (conveyor_id % 2)),
                        (1 - conveyor_id // 2) * (1 - 2 * (conveyor_id % 2))]
        # If the conveyor moves player character off the grid or to a wall,
        # it breaks the loop and does not add the direction to the final player direction.
        if check_player_move_off_grid(player, pdir[0] + conveyor_dir[0], pdir[1] + conveyor_dir[1], matrix, p1_pos, p2_pos)\
                or matrix[ppos[0] + pdir[1] + conveyor_dir[1]][ppos[1] + pdir[0] + conveyor_dir[0]] == 4:
            break
        pdir = [pdir[0] + conveyor_dir[0], pdir[1] + conveyor_dir[1]]

    return pdir

def move_players(x, y, matrix, p1_pos, p2_pos, layout):
    """
        Moves the player characters.
    """
    # Checks if there are valid moves, returns None otherwise.
    if not check_moves(x, y, matrix, p1_pos, p2_pos):
        return

    MATRIX, P1, P2 = copy.deepcopy(matrix), copy.deepcopy(p1_pos), copy.deepcopy(p2_pos)

    # Gets the player character 1 and player character 2's direction.
    p1_dir = get_player_direction(1, x, y, MATRIX, P1, P2)
    p2_dir = get_player_direction(2, x, y, MATRIX, P1, P2)

    # If the player character 1 and player character 2 try to move to the same cell, it merges them.
    if p1_pos[0] + p1_dir[1] == p2_pos[0] + p2_dir[1] and \
            p1_pos[1] + p1_dir[0] == p2_pos[1] + p2_dir[0]:
        MATRIX[p1_pos[0]][p1_pos[1]] = layout[p1_pos[0]][p1_pos[1]]
        MATRIX[p2_pos[0]][p2_pos[1]] = layout[p2_pos[0]][p2_pos[1]]
        MATRIX[p1_pos[0] + p1_dir[1]][p1_pos[1] + p1_dir[0]] = 3
    # If the player character 1 and player character 2 try to move to each other's cell,
    # it merges them in the place of player character 2.
    elif P1[0] + p1_dir[1] == P2[0] and P1[1] + p1_dir[0] == P2[1] \
            and P2[0] + p2_dir[1] == P1[0] and P2[1] + p2_dir[0] == P1[1]:
        MATRIX[P1[0]][P1[1]] = layout[P1[0]][P1[1]]
        MATRIX[P2[0]][P2[1]] = 3
    # If the player character 1 and player character 2 does not interact with each other, it performs the move.
    else:
        # If the player character 1 has a valid move, it moves the player character 1
        # and updates the previous position of the player character 1 with the original tile of the layout.
        if p1_dir[0] != 0 or p1_dir[1] != 0:
            MATRIX[P1[0]][P1[1]] = layout[P1[0]][P1[1]]
            P1 = (P1[0] + p1_dir[1], P1[1] + p1_dir[0])
        # If the player character 2 has a valid move, it moves the player character 2
        # and updates the previous position of the player character 2 with the original tile of the layout.
        if p2_dir[0] != 0 or p2_dir[1] != 0:
            MATRIX[P2[0]][P2[1]] = layout[P2[0]][P2[1]]
            P2 = (P2[0] + p2_dir[1], P2[1] + p2_dir[0])
        # Changes the target cells of movement to the player character 1 and player character 2.
        MATRIX[P1[0]][P1[1]] = 1
        MATRIX[P2[0]][P2[1]] = 2
    return MATRIX

def check_win(matrix):
    """
        Checks if the game is won.
    """
    for row in matrix:
        for cell in row:
            if cell == 3:
                return True
    return False

class BFSAgent(Agent):
    def __init__(self, matrix):
        """
            Initializes the BFS agent class.

            Args:
                matrix (list): Initial game matrix
        """
        # Initializing the parent class
        super().__init__(matrix)

    def tree_solve(self):
        """
            Solves the game using tree-based BFS algorithm.

            Returns:
                list: A list of tuples containing moves and state matrices in the solution.
        """
        ####################################
        ######## YOUR SOLUTION HERE ########
        ####################################

        limit = 4 # Time limit for BFS algorithm to solve
        started_at = time.time() # Starting time

        # Deep copy layout
        layout = copy.deepcopy(self.initial_matrix) 
        p1_pos, p2_pos = self.find_players_positions(self.initial_matrix)
        layout[p1_pos[0]][p1_pos[1]] = 0
        layout[p2_pos[0]][p2_pos[1]] = 0

        # Will use a list like a FIFO queue
        self.frontier = []

        # Create a node from initial matrix and push into frontier
        root = Node(None, self.initial_matrix, move_before=None)
        self.frontier.append(root)
        self.generated_node += 1

        self.maximum_node_in_memory = len(self.frontier)

        while self.frontier: # While frontiner is not empty
            # Pop a node from frontier
            node = self.frontier[0]
            self.frontier = self.frontier[1:] # Remove popped node from the list
            self.explored_node += 1

            # If time exceeded the limit to find a solution return the moves of the current node
            if time.time() - started_at > limit:
                print("Time limit exceeded. BFS AGENT could not find a solution.")
                return self.get_moves(node)

            # Check if the current node is winning condition, if so return the moves 
            if check_win(node.matrix):
                return self.get_moves(node)
            
            # Get the positions of players [x1, y1], [x2, y2]
            p1, p2 = self.find_players_positions(node.matrix)
            
            for move in self.actions: # Iterate over defined moves for state transitions (DOWN, LEFT, UP, RIGHT)
                if check_moves(move[0], move[1], node.matrix, p1, p2): # If move is valid:
                    new_matrix = move_players(move[0], move[1], node.matrix, p1, p2, layout) # Proceed to move and create changed matrix
                    
                    # If move is valid but the matrix is same with previous matrix skip this iteration
                    if self.check_equal(node.matrix, new_matrix):
                        continue

                    self.total_move += 1

                    # Create a node with this move and push to frontier
                    self.generated_node += 1
                    front = Node(node, new_matrix, move)
                    self.frontier.append(front)

                    self.maximum_node_in_memory = max(self.maximum_node_in_memory, len(self.frontier))

        return []

    def graph_solve(self):
        """
            Solves the game using graph-based BFS algorithm.

            Returns:
                list: A list of tuples containing moves and state matrices in the solution.
        """
        ####################################
        ######## YOUR SOLUTION HERE ########
        ####################################

        # Deep copy layout
        layout = copy.deepcopy(self.initial_matrix)
        p1_pos, p2_pos = self.find_players_positions(self.initial_matrix)
        layout[p1_pos[0]][p1_pos[1]] = 0
        layout[p2_pos[0]][p2_pos[1]] = 0

        # Will use a list like a FIFO queue
        self.frontier = []

        # A set to keep track of matrices of seen states
        self.explored = set()

        # Create a node from initial matrix and push into frontier and add the matrix to explored set
        root = Node(None, self.initial_matrix, move_before=None)
        self.frontier.append(root)
        self.explored.add(matrix_to_tuple(self.initial_matrix))
        self.generated_node += 1

        self.maximum_node_in_memory = len(self.frontier) + len(self.explored)

        while self.frontier: # While frontiner is not empty
            # Pop a node from frontier
            node = self.frontier[0]
            self.frontier = self.frontier[1:] # Remove popped node from the list
            self.explored_node += 1
            
            # Check if the current node is winning condition, if so return the moves 
            if check_win(node.matrix):
                return self.get_moves(node)
            
            # Get the positions of players [x1, y1], [x2, y2]
            p1, p2 = self.find_players_positions(node.matrix)
            
            for move in self.actions: # Iterate over defined moves for state transitions (DOWN, LEFT, UP, RIGHT)
                if check_moves(move[0], move[1], node.matrix, p1, p2): # If move is valid:
                    new_matrix = move_players(move[0], move[1], node.matrix, p1, p2, layout) # Proceed to move and create changed matrix

                    # If move is valid but the matrix is same with previous matrix skip this iteration
                    if self.check_equal(node.matrix, new_matrix):
                        continue

                    self.total_move += 1

                    # Create a matrix with set(set) structure and search it in explored set
                    new_matrix_tuple = matrix_to_tuple(new_matrix)
                    if new_matrix_tuple in self.explored: # If found skip this iteration 
                        continue

                    # Create a node with this move and push to frontier and add the matrix to explored set
                    self.generated_node += 1
                    front = Node(node, new_matrix, move)
                    self.frontier.append(front)
                    self.explored.add(new_matrix_tuple)

                    self.maximum_node_in_memory = max(self.maximum_node_in_memory, len(self.frontier) + len(self.explored))

        return []

class DFSAgent(Agent):
    def __init__(self, matrix):
        """
            Initializes the DFS agent class.

            Args:
                matrix (list): Initial game matrix
        """
        # Initializing the parent class
        super().__init__(matrix)

    def tree_solve(self):
        """
            Solves the game using tree-based DFS algorithm.

            Returns:
                list: A list of tuples containing moves and state matrices in the solution.
        """
        ####################################
        ######## YOUR SOLUTION HERE ########
        ####################################

        limit = 4 # Time limit for DFS algorithm to solve
        started_at = time.time() # Starting time

        # Deep copy layout
        layout = copy.deepcopy(self.initial_matrix)
        p1_pos, p2_pos = self.find_players_positions(self.initial_matrix)
        layout[p1_pos[0]][p1_pos[1]] = 0
        layout[p2_pos[0]][p2_pos[1]] = 0

        # Will use a list like a LIFO queue (stack)
        self.frontier = []

        # Create a node from initial matrix and push into frontier
        root = Node(None, self.initial_matrix, move_before=None)
        self.frontier.append(root)
        self.generated_node += 1

        self.maximum_node_in_memory = len(self.frontier)

        while self.frontier: # While frontiner is not empty
            # Pop a node from frontier
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1] # Remove popped node from the list
            self.explored_node += 1

            # If time exceeded the limit to find a solution return the moves of the current node
            if time.time() - started_at > limit:
                print("Time limit exceeded. DFS AGENT could not find a solution.")
                return self.get_moves(node)
            
            # Check if the current node is winning condition, if so return the moves 
            if check_win(node.matrix):
                return self.get_moves(node)
            
            # Get the positions of players [x1, y1], [x2, y2]
            p1, p2 = self.find_players_positions(node.matrix)
            
            for move in self.actions: # Iterate over defined moves for state transitions (DOWN, LEFT, UP, RIGHT)
                if check_moves(move[0], move[1], node.matrix, p1, p2): # If move is valid:
                    new_matrix = move_players(move[0], move[1], node.matrix, p1, p2, layout) # Proceed to move and create changed matrix

                    # If move is valid but the matrix is same with previous matrix skip this iteration
                    if self.check_equal(node.matrix, new_matrix):
                        continue

                    self.total_move += 1

                    # Create a node with this move and push to frontier
                    self.generated_node += 1
                    front = Node(node, new_matrix, move)
                    self.frontier.append(front)

                    self.maximum_node_in_memory = max(self.maximum_node_in_memory, len(self.frontier))

        return []

    def graph_solve(self):
        """
            Solves the game using graph-based DFS algorithm.

            Returns:
                list: A list of tuples containing moves and state matrices in the solution.
        """
        ####################################
        ######## YOUR SOLUTION HERE ########
        ####################################

        # Deep copy layout
        layout = copy.deepcopy(self.initial_matrix)
        p1_pos, p2_pos = self.find_players_positions(self.initial_matrix)
        layout[p1_pos[0]][p1_pos[1]] = 0
        layout[p2_pos[0]][p2_pos[1]] = 0

        # Will use a list like a LIFO queue (stack)
        self.frontier = []

        # A set to keep track of matrices of seen states
        self.explored = set()

        # Create a node from initial matrix and push into frontier and add the matrix to explored set
        root = Node(None, self.initial_matrix, move_before=None)
        self.frontier.append(root)
        self.explored.add(matrix_to_tuple(root.matrix))
        self.generated_node += 1

        self.maximum_node_in_memory = len(self.frontier) + len(self.explored)

        while self.frontier: # While frontiner is not empty
            # Pop a node from frontier
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1] # Remove popped node from the list
            self.explored_node += 1
            
            # Check if the current node is winning condition, if so return the moves 
            if check_win(node.matrix):
                return self.get_moves(node)
            
            # Get the positions of players [x1, y1], [x2, y2]
            p1, p2 = self.find_players_positions(node.matrix)
            
            for move in self.actions: # Iterate over defined moves for state transitions (DOWN, LEFT, UP, RIGHT)
                if check_moves(move[0], move[1], node.matrix, p1, p2): # If move is valid:
                    new_matrix = move_players(move[0], move[1], node.matrix, p1, p2, layout) # Proceed to move and create changed matrix

                    # If move is valid but the matrix is same with previous matrix skip this iteration
                    if self.check_equal(node.matrix, new_matrix):
                        continue

                    self.total_move += 1

                    # Create a matrix with set(set) structure and search it in explored set
                    new_matrix_tuple = matrix_to_tuple(new_matrix)
                    if new_matrix_tuple in self.explored:
                        continue

                    # Create a node with this move and push to frontier and add the matrix to explored set
                    self.generated_node += 1
                    front = Node(node, new_matrix, move)
                    self.frontier.append(front)
                    self.explored.add(new_matrix_tuple)

                    self.maximum_node_in_memory = max(self.maximum_node_in_memory, len(self.frontier) + len(self.explored))

        return []

class AStarAgent(Agent):
    def __init__(self, matrix):
        """
            Initializes the A* agent class.

            Args:
                matrix (list): Initial game matrix
        """
        # Initializing the parent class
        super().__init__(matrix)

    def manhattan_distance(self, matrix):
        """
            Finds the locations of each player and returns the Manhattan distance between two players calculated by h(x1, y1, x2, y2) = |x1 - x2| + |y1 - y2|.

            Args:
                matrix (list): Initial game matrix (2x2)

            Returns:
                int: Manhattan distance calculated between 2 players: h(x1, y1, x2, y2) = |x1 - x2| + |y1 - y2|
        """
        try:
            p1_pos, p2_pos = self.find_players_positions(matrix)
            return abs(p1_pos[0] - p2_pos[0]) + abs(p1_pos[1] - p2_pos[1])
        
        except TypeError:
            return 0
        
    def chebyshev_distance(self, matrix):
        """
            Finds the locations of each player and returns the Chebyshev distance between two players calculated by h(x1, y1, x2, y2) = max(|x1 - x2|, |y1 - y2|).

            Args:
                matrix (list): Initial game matrix (2x2)

            Returns:
                int: Chebyshev distance calculated between 2 players: h(x1, y1, x2, y2) = max(|x1 - x2|, |y1 - y2|)
        """
        try:
            p1_pos, p2_pos = self.find_players_positions(matrix)
            return max(abs(p1_pos[0] - p2_pos[0]), abs(p1_pos[1] - p2_pos[1]))
        
        except TypeError:
            return 0
        
    def euclidean_distance(self, matrix):
        """
            Finds the locations of each player and returns the Euclidean distance between two players calculated by h(x1, y1, x2, y2) = sqrt((x1 - x2)^2 + (y1 - y2)^2).

            Args:
                matrix (list): Initial game matrix (2x2)

            Returns:
                float: Euclidean distance calculated between 2 players: h(x1, y1, x2, y2) = sqrt((x1 - x2)^2 + (y1 - y2)^2)
        """
        try:
            p1_pos, p2_pos = self.find_players_positions(matrix)
            return ((p1_pos[0] - p2_pos[0]) ** 2 + (p1_pos[1] - p2_pos[1]) ** 2) ** 0.5
        
        except TypeError:
            return 0

    def tree_solve(self):
        """
            Solves the game using tree-based A* algorithm.

            Returns:
                list: A list of tuples containing moves and state matrices in the solution.
        """
        ####################################
        ######## YOUR SOLUTION HERE ########
        ####################################

        limit = 4 # Time limit for A* algorithm to solve
        started_at = time.time() # Starting time

        # Deep copy layout
        layout = copy.deepcopy(self.initial_matrix)
        p1_pos, p2_pos = self.find_players_positions(self.initial_matrix)
        layout[p1_pos[0]][p1_pos[1]] = 0
        layout[p2_pos[0]][p2_pos[1]] = 0

        # Will use a priority queue for frontier, priority is the total cost of the state
        self.frontier = PriorityQueue()

        # Create a node from initial matrix and push into frontier with 0 initial cost and manhattan distance for heuristic cost
        root = Node(None, self.initial_matrix, move_before=None, g_score=0, h_score=self.euclidean_distance(self.initial_matrix))
        self.frontier.push(root, root.f_score)
        self.generated_node += 1

        self.maximum_node_in_memory = self.frontier.size()

        while self.frontier.size(): # While frontiner is not empty
            node = self.frontier.pop() # Pop node with lowest cost state from the frontier
            self.explored_node += 1
            
            # If time exceeded the limit to find a solution return the moves of the current node
            if time.time() - started_at > limit:
                print("Time limit exceeded. A* AGENT could not find a solution.")
                return self.get_moves(node)

            # Check if the current node is winning condition, if so return the moves 
            if check_win(node.matrix):
                return self.get_moves(node)
            
            # Get the positions of players [x1, y1], [x2, y2]
            p1, p2 = self.find_players_positions(node.matrix)
            
            for move in self.actions: # Iterate over defined moves for state transitions (DOWN, LEFT, UP, RIGHT)
                if check_moves(move[0], move[1], node.matrix, p1, p2): # If move is valid:
                    new_matrix = move_players(move[0], move[1], node.matrix, p1, p2, layout) # Proceed to move and create changed matrix
                    
                    # If move is valid but the matrix is same with previous matrix skip this iteration
                    if self.check_equal(node.matrix, new_matrix):
                        continue

                    self.total_move += 1

                    # Create a node with this move and push to frontier with g(x) and h(x)
                    self.generated_node += 1
                    front = Node(parent=node, matrix=new_matrix, move_before=move, g_score=node.g_score+1, h_score=self.euclidean_distance(new_matrix))
                    self.frontier.push(front, front.f_score)

                    self.maximum_node_in_memory = max(self.maximum_node_in_memory, self.frontier.size())

        return []

    def graph_solve(self):
        """
            Solves the game using graph-based A* algorithm.

            Returns:
                list: A list of tuples containing moves and state matrices in the solution.
        """
        ####################################
        ######## YOUR SOLUTION HERE ########
        ####################################

        # Deep copy layout
        layout = copy.deepcopy(self.initial_matrix)
        p1_pos, p2_pos = self.find_players_positions(self.initial_matrix)
        layout[p1_pos[0]][p1_pos[1]] = 0
        layout[p2_pos[0]][p2_pos[1]] = 0

        # Will use a priority queue for frontier, priority is the total cost of the state
        self.frontier = PriorityQueue()

        # A set to keep track of matrices of seen states
        self.explored = set()

        # Create a node from initial matrix and push into frontier with 0 initial cost and manhattan distance for heuristic cost
        root = Node(None, self.initial_matrix, move_before=None, g_score=0, h_score=self.euclidean_distance(self.initial_matrix))
        self.frontier.push(root, root.f_score)
        self.explored.add(matrix_to_tuple(root.matrix))
        self.generated_node += 1

        self.maximum_node_in_memory = self.frontier.size() + len(self.explored)

        while self.frontier.size(): # While frontiner is not empty
            node = self.frontier.pop() # Pop node with lowest cost state from the frontier
            self.explored_node += 1

            # Check if the current node is winning condition, if so return the moves 
            if check_win(node.matrix):
                return self.get_moves(node)
            
            # Get the positions of players [x1, y1], [x2, y2]
            p1, p2 = self.find_players_positions(node.matrix)

            for move in self.actions: # Iterate over defined moves for state transitions (DOWN, LEFT, UP, RIGHT)
                if check_moves(move[0], move[1], node.matrix, p1, p2):
                    new_matrix = move_players(move[0], move[1], node.matrix, p1, p2, layout) # Proceed to move and create changed matrix

                    # If move is valid but the matrix is same with previous matrix skip this iteration
                    if self.check_equal(node.matrix, new_matrix):
                        continue

                    self.total_move += 1

                    # Create a matrix with set(set) structure and search it in explored set
                    new_matrix_tuple = matrix_to_tuple(new_matrix)
                    if new_matrix_tuple in self.explored:
                        continue

                    # Create a node with this move and push to frontier with g(x) and h(x) and add the matrix to explored set
                    self.generated_node += 1
                    front = Node(parent=node, matrix=new_matrix, move_before=move, g_score=node.g_score+1, h_score=self.euclidean_distance(new_matrix))
                    self.frontier.push(front, front.f_score)
                    self.explored.add(new_matrix_tuple)

                    self.maximum_node_in_memory = max(self.maximum_node_in_memory, self.frontier.size() + len(self.explored))
        
        return []


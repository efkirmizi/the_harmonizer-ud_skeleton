import copy
import heapq


class Agent:
    def __init__(self, matrix):
        """
            Initializes the agent class.

            Args:
                matrix (list): Initial game matrix
        """
        # The initial game matrix
        self.initial_matrix = [list(row) for row in matrix]

        # The desired value for goal state
        self.desired_value = 3

        # Large value to represent infinity
        self.INFINITY = 2 ** 32

        # Player values
        self.player1 = 1
        self.player2 = 2

        # Action vectors for directions DOWN, LEFT, UP, RIGHT
        self.actions = [[0, 1], [-1, 0], [0, -1], [1, 0]]

        # The frontier and explored sets for the search algorithms 
        self.frontier = None
        self.explored = None

        # Information regarding nodes
        self.explored_node = 0
        self.generated_node = 0
        self.maximum_node_in_memory = 0

        # The total number of moves
        self.total_move = 0

    def find_players_positions(self, matrix):
        """
            Returns the positions of the players on the current game matrix.

            Args:
                matrix (list): The current game matrix

            Returns:
                list (list(int), list(int)): The positions of the players
        """
        p1_pos = [-1, -1]
        p2_pos = [-1, -1]
        for i, row in enumerate(matrix):
            for j, cell in enumerate(row):
                if cell == self.player1 or cell == self.player2:
                    if cell == self.player1:
                        p1_pos = [i, j]
                    elif cell == self.player2:
                        p2_pos = [i, j]
                    if p1_pos != [-1, -1] and p2_pos != [-1, -1]:
                        return p1_pos, p2_pos

    def get_moves(self, node):
        """
            Returns a list of tuples of moves and game state matrices that represents the solution.

            Args:
                node (Node): Node that will initialize the solution path

            Returns:
                list: A list of game matrices
        """
        moves = []
        road = []
        while node.parent is not None:
            if node.move_before is not None:
                moves.append(node.move_before)
            road.append(node.matrix)
            node = node.parent
            self.total_move += 1
        return moves[::-1], road[::-1]

    def print_info(self):
        """
            Prints the information about the agent.
        """
        print("The Generated Node Count: " + str(self.generated_node))
        print("The Expanded Node Count: " + str(self.explored_node))
        print("The Maximum Node Count in the Memery: " + str(self.maximum_node_in_memory))

    @staticmethod
    def check_value(matrix, value):
        """
            Checks if the value is in the matrix.

            Args:
                matrix (list): The matrix to be looked into
                value (int): The value to be checked
            Returns:
                bool: True if the matrix contains a cell with the value, False otherwise
        """
        for row in matrix:
            for cell in row:
                if cell == value:
                    return True
        return False

    @staticmethod
    def check_equal(matrix1, matrix2):
        """
            Checks if two matrices are equal.

            Args:
                matrix1 (list): The first matrix to be compared
                matrix2 (list): The second matrix to be compared
            Returns:
                bool: True if the matrices are equal, False otherwise
        """
        for row1, row2 in zip(matrix1, matrix2):
            for cell1, cell2 in zip(row1, row2):
                if cell1 != cell2:
                    return False
        return True

    @staticmethod
    def copy_matrix(matrix):
        """
            Copies the given matrix.

            Args:
                matrix (list): The matrix to be copied

            Returns:
                list: The copied matrix
        """
        return copy.deepcopy(matrix)


class Node:
    def __init__(self, parent, matrix, move_before, g_score=0, h_score=0):
        """
            Initializes the node class.

            Args:
                parent (Node): Parent node
                matrix (list): Game matrix
                move_before (list): The performed move to get this state
                g_score (int): G score of the node
                h_score (int): H score of the node
        """
        self.parent = parent
        self.matrix = matrix
        self.move_before = move_before

        self.g_score = g_score
        self.h_score = h_score
        self.f_score = g_score + h_score

    def __lt__(self, other):
        """
            Compares the g and h scores of two nodes.

            Args:
                other (Node): Other node to compare
        """
        return self.f_score < other.f_score


class PriorityQueue:

    def __init__(self):
        """
            Initializes the priority queue class.
        """
        self.elements = []

    def is_empty(self):
        """
            Checks if the priority queue is empty.

            Returns:
                bool: True if the priority queue is empty, False otherwise
        """
        return len(self.elements) == 0

    def push(self, item, priority):
        """
            Pushes an item into the priority queue.

            Args:
                item (Node): Item to be pushed into the priority queue
                priority (int): Priority of the item
        """
        heapq.heappush(self.elements, (priority, item))

    def pop(self):
        """
            Pops an item from the priority queue.

            Returns:
                Node: Item with the highest priority
        """
        return heapq.heappop(self.elements)[1]

    def size(self):
        """
            Returns the size of the priority queue.

            Returns:
                int: Size of the priority queue
        """
        return len(self.elements)

    def contains(self, matrix):
        """
            Checks if the priority queue contains the given matrix.

            Args:
                matrix (list): Matrix to be checked

            Returns:   
                bool: True if the priority queue contains the matrix, False otherwise
        """
        for element in self.elements:
            if element[1].matrix == matrix:
                return True
        return False

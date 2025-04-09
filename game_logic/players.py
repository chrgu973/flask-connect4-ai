from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import random
import math
import time
import copy
import pickle
import os

from .board import (
    create_board, print_board, is_valid_location, get_next_open_row,
    drop_piece, get_valid_locations, winning_move, is_terminal_node,
    PLAYER1_PIECE, PLAYER2_PIECE, EMPTY, ROWS, COLS,
    POSITIONAL_VALUES 
)

############################################################################
################# Player base class ########################################
############################################################################

class Player(ABC):
    """Abstract base class for all Connect 4 players."""
    def __init__(self, player_id):
        self.player_id = player_id 

    @abstractmethod
    def get_move(self, board):
        """
        Given the current board state, returns the column where the player wants to move.

        Args:
            board (np.ndarray): The current 6x7 game board.

        Returns:
            int: The column index (0-6) for the move.
        """
        pass

class HumanPlayer(Player):
    """A player controlled by human input via the console (accepts 1-7)."""
    def get_move(self, board):
        """Gets move from user input, expecting 1-7."""
        # Get valid locations based on the current board state (0-based indices)
        valid_locations_zero_based = get_valid_locations(board)
        
        # Convert to 1-based for display to users (1-7 for columns instead of 0-6)
        valid_locations_display = [loc + 1 for loc in valid_locations_zero_based]  # Convert to 1-based for display

        # Check if there are any valid moves available
        if not valid_locations_display:
             print("Error: No valid moves available!")
             return None 

        # Keep asking for input until a valid move is provided
        while True:
            try:
                # Get column selection from user (1-7)
                col_str = input(f"Player {self.player_id}, choose column ({', '.join(map(str, valid_locations_display))}): ")
                user_col = int(col_str) 

                # Convert from 1-based display to 0-based internal
                internal_col = user_col - 1  

                # Verify the move is valid
                if internal_col in valid_locations_zero_based:
                    return internal_col 
                else:
                    print(f"Invalid column {user_col}. Please choose from {valid_locations_display}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")


############################################################################
################# CNNMinimaxPlayer #########################################
############################################################################

class CNNMinimaxPlayer(Player):
    def __init__(self, player_id, model_path='connect4_cnn_model.h5', search_depth=4):
        """
        Initializes the AI player.

        Args:
            player_id (int): PLAYER1_PIECE or PLAYER2_PIECE.
            model_path (str): Path to the saved Keras model file.
            search_depth (int): The depth for the Minimax search.
        """
        super().__init__(player_id)
        # Store opponent ID for evaluations
        self.opponent_id = PLAYER1_PIECE if player_id == PLAYER2_PIECE else PLAYER2_PIECE
        self.search_depth = search_depth
        try:
            # Load the CNN model for board evaluation
            self.model = tf.keras.models.load_model(model_path)
            print(f"CNN Model loaded successfully from {model_path} for Player {player_id}")
            # Test model with a dummy prediction to ensure it's ready
            dummy_board = create_board().reshape(1, ROWS, COLS, 1)
            _ = self.model.predict(dummy_board, verbose=0) 
            print("Model ready.")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("CNNMinimaxPlayer will not function correctly.")
            self.model = None

    def _evaluate_board_cnn(self, board):
        """
        Evaluates the board state using the CNN model.
        Returns a score from Player 1's perspective (higher is better for P1).
        """
        if self.model is None:
            return 0 # Cannot evaluate without a model
        
        # Reshape board for CNN input (adds batch and channel dimensions)
        board_cnn = board.reshape(1, ROWS, COLS, 1)
        # Get model prediction (win probability for each outcome)
        probabilities = self.model.predict(board_cnn, verbose=0)[0]

        # Calculate score as: P1 win probability - P2 win probability
        score = probabilities[1] - probabilities[2]  
        return score

    def _minimax(self, board, depth, maximizing_player, alpha, beta):
        """
        Minimax algorithm with Alpha-Beta pruning.

        Returns:
            tuple: (column, score) - Score is from Player 1's perspective.
        """
        # Get valid moves for current board state
        valid_locations = get_valid_locations(board)
        is_terminal = is_terminal_node(board)

        # Base cases: reached depth limit or terminal node
        if depth == 0 or is_terminal:
            if is_terminal:
                if winning_move(board, PLAYER1_PIECE):
                    return (None, 1000000 + depth) # Prioritize faster wins
                elif winning_move(board, PLAYER2_PIECE):
                    return (None, -1000000 - depth) # Prioritize blocking faster losses
                else: # Game is draw
                    return (None, 0)
            else: # Depth is zero, use CNN evaluation
                return (None, self._evaluate_board_cnn(board))

        if maximizing_player: # Player 1's turn (Maximize score)
            value = -math.inf
            best_col = random.choice(valid_locations) # Default move
            for col in valid_locations:
                # Simulate the move
                row = get_next_open_row(board, col)
                temp_board = board.copy()
                drop_piece(temp_board, row, col, PLAYER1_PIECE)
                # Recursive call with reduced depth
                _, new_score = self._minimax(temp_board, depth - 1, False, alpha, beta)
                if new_score > value:
                    value = new_score
                    best_col = col
                alpha = max(alpha, value)
                # Alpha-beta pruning
                if alpha >= beta:
                    break # Beta cutoff
            return best_col, value
        else: # Player 2's turn (Minimize score from P1's perspective)
            value = math.inf
            best_col = random.choice(valid_locations) # Default move
            for col in valid_locations:
                # Simulate the move
                row = get_next_open_row(board, col)
                temp_board = board.copy()
                drop_piece(temp_board, row, col, PLAYER2_PIECE)
                # Recursive call with reduced depth
                _, new_score = self._minimax(temp_board, depth - 1, True, alpha, beta)
                if new_score < value:
                    value = new_score
                    best_col = col
                # Alpha-beta pruning
                beta = min(beta, value)
                if alpha >= beta:
                    break # Alpha cutoff
            return best_col, value

    def get_move(self, board):
        """
        Determines the AI's move using Minimax with the CNN evaluator.
        Includes checking for immediate wins and blocking opponent wins.
        """
        if self.model is None:
             print("AI Error: Model not loaded. Choosing random move.")
             return random.choice(get_valid_locations(board))

        valid_locations = get_valid_locations(board)
        start_time = time.time()

        # 1. Check for immediate winning move for self
        for col in valid_locations:
            temp_board = board.copy()
            row = get_next_open_row(temp_board, col)
            drop_piece(temp_board, row, col, self.player_id)
            if winning_move(temp_board, self.player_id):
                print(f"AI Player {self.player_id}: Found winning move in column {col}")
                return col

        # 2. Check for immediate winning move for opponent and block it
        for col in valid_locations:
            temp_board = board.copy()
            row = get_next_open_row(temp_board, col)
            drop_piece(temp_board, row, col, self.opponent_id)
            if winning_move(temp_board, self.opponent_id):
                print(f"AI Player {self.player_id}: Blocking opponent win in column {col}")
                return col

        # 3. If no immediate win/block, use Minimax
        print(f"AI Player {self.player_id}: Running Minimax (depth {self.search_depth})...")
        col, minimax_score = self._minimax(board, self.search_depth, self.player_id == PLAYER1_PIECE, -math.inf, math.inf)

        end_time = time.time()
        print(f"AI Player {self.player_id}: Chose column {col} (Score: {minimax_score:.2f}, Time: {end_time - start_time:.2f}s)")

        # Sanity check in case minimax fails
        if col is None or col not in valid_locations: # Fallback if minimax fails
             print(f"AI Warning: Minimax returned invalid move {col}. Choosing random valid move.")
             col = random.choice(valid_locations)

        return col
    

############################################################################
################# Random players ###########################################
############################################################################


class RandomAIPlayer(Player):
    """An AI player that chooses a valid move randomly."""
    def get_move(self, board):
        # Get valid moves and pick one randomly
        valid_locations = get_valid_locations(board)
        move = random.choice(valid_locations)
        print(f"Random AI Player {self.player_id}: Chose column {move}")
        time.sleep(0.5) # Add a small delay to simulate thinking
        return move    


class SlightlyBetterRandomAIPlayer(Player):
    """
    An AI player that makes moves based on the following priority:
    1. Play a winning move if available.
    2. Block the opponent's winning move if available.
    3. Choose a random valid move otherwise.
    """
    def __init__(self, player_id):
        """
        Initializes the player.

        Args:
            player_id (int): PLAYER1_PIECE or PLAYER2_PIECE.
        """
        super().__init__(player_id)
        # Store opponent's ID for checking blocking moves
        self.opponent_id = PLAYER1_PIECE if player_id == PLAYER2_PIECE else PLAYER2_PIECE
        print(f"SlightlyBetterRandomAIPlayer initialized for Player {self.player_id} (Opponent: {self.opponent_id})")

    def get_move(self, board):
        """
        Determines the move based on win, block, or random choice.

        Args:
            board (np.ndarray): The current 6x7 game board.

        Returns:
            int: The column index (0-6) for the move.
        """
        valid_locations = get_valid_locations(board)

        # 1. Check for immediate winning move for self
        for col in valid_locations:
            temp_board = board.copy() # Use a copy to simulate the move
            row = get_next_open_row(temp_board, col)
            if row is not None: # Ensure the column wasn't full (should be covered by valid_locations)
                drop_piece(temp_board, row, col, self.player_id)
                if winning_move(temp_board, self.player_id):
                    return col

        # 2. Check for immediate winning move for opponent and block it
        for col in valid_locations:
            temp_board = board.copy() # Use a copy to simulate opponent's move
            row = get_next_open_row(temp_board, col)
            if row is not None:
                drop_piece(temp_board, row, col, self.opponent_id)
                if winning_move(temp_board, self.opponent_id):
                    return col

        # 3. If no win/block, choose a random valid move
        move = random.choice(valid_locations)
        return move



############################################################################
################# MCTS player ##############################################
############################################################################
    

class MCTSNode:
    """ Represents a node in the Monte Carlo Search Tree. """
    def __init__(self, state, parent=None, move=None, player_at_node=None):
        self.state = state  # The board state at this node
        self.parent = parent  # Reference to parent node
        self.move = move  # The move that led to this node (column index)
        self.children = []  # Child nodes
        self.visits = 0  # Number of times this node was visited
        self.score = 0  # Cumulative score for this node 
        self.untried_moves = get_valid_locations(state)  # Valid moves not yet expanded
        self.player_at_node = player_at_node # Player whose turn it is AT THIS NODE

    def uct_select_child(self, exploration_constant=1.414):
        """ Selects a child node using the UCT formula, from the parent's perspective. """
        # Get children with at least one visit
        children_with_visits = [c for c in self.children if c.visits > 0]

        if not self.children:
             return None 

        # Ensure parent has visits for log calculation
        if self.visits == 0:
             return random.choice(self.children) if self.children else None

        log_parent_visits = math.log(self.visits)

        def uct_score(node):
            """ Calculates the UCT score for a child node from the parent's perspective. """
            if node.visits == 0:
                return float('inf')  # Ensures unvisited nodes are tried first

            # Win rate from child's perspective
            child_player_win_rate = node.score / node.visits

            # Invert for parent's perspective (parent wants the opposite outcome)
            parent_perspective_win_rate = -child_player_win_rate

            # UCT exploration term
            exploration_term = exploration_constant * math.sqrt(log_parent_visits / node.visits)

            # Complete UCT formula: exploitation + exploration
            return parent_perspective_win_rate + exploration_term

        # Select child with highest UCT score
        selected_child = max(self.children, key=uct_score)

        return selected_child

    def add_child(self, move, state, player_at_new_node):
        """ Adds a new child node. """
        # Create new child node
        node = MCTSNode(state=state, parent=self, move=move, player_at_node=player_at_new_node)
        # Remove the move from untried moves
        if move in self.untried_moves:
             self.untried_moves.remove(move)
        # Add to children list
        self.children.append(node)
        return node

    def update(self, result_from_perspective_of_player_at_this_node):
        """ Updates visit count and score. """
        self.visits += 1
        # Update score based on the result from this node's player perspective
        self.score += result_from_perspective_of_player_at_this_node


class MCTSPlayer(Player):
    """ AI player implementing Monte Carlo Tree Search with Heuristic Playouts. """
    def __init__(self, player_id, iterations=1000, exploration_constant=1.414):
        super().__init__(player_id)
        self.opponent_id = PLAYER1_PIECE if player_id == PLAYER2_PIECE else PLAYER2_PIECE
        self.n_iterations = iterations  # Number of MCTS iterations per move
        self.exploration_constant = exploration_constant  # UCT exploration parameter
        self.positional_values = POSITIONAL_VALUES  # Board position heuristics
        print(f"MCTSPlayer initialized for Player {self.player_id} ({self.n_iterations} iterations/move, Heuristic Playouts, Corrected UCT)")

    def get_move(self, board):
        start_time = time.time()
        # Create root node with current board state
        root = MCTSNode(state=board.copy(), player_at_node=self.player_id)

        # Check immediate win/loss first
        valid_locations = get_valid_locations(board)
        # 1. Look for winning move
        for col in valid_locations:
             temp_board_win = board.copy()
             row_win = get_next_open_row(temp_board_win, col)
             drop_piece(temp_board_win, row_win, col, self.player_id)
             if winning_move(temp_board_win, self.player_id):
                 print(f"MCTS Player {self.player_id}: Found immediate winning move {col}")
                 return col
        # 2. Look for blocking move
        for col in valid_locations:
             temp_board_loss = board.copy()
             row_loss = get_next_open_row(temp_board_loss, col)
             drop_piece(temp_board_loss, row_loss, col, self.opponent_id)
             if winning_move(temp_board_loss, self.opponent_id):
                 print(f"MCTS Player {self.player_id}: Found immediate block at {col}")
                 return col

        # MCTS main loop
        for i in range(self.n_iterations):
            node = root
            current_board_state = board.copy()

            # 1. Selection - traverse tree until we reach a leaf node
            while not node.untried_moves and node.children:
                node = node.uct_select_child(self.exploration_constant)
                if node is None: break # Safety check if no children selectable
                # Apply move to descend tree
                row = get_next_open_row(current_board_state, node.move)
                if row is None:
                    print(f"Warning: Invalid move {node.move} selected during descent.")
                    break
                # The parent node's player made this move
                drop_piece(current_board_state, row, node.move, node.parent.player_at_node) # Player at parent made the move

            if node is None: continue 

            # 2. Expansion - create a new child node if not terminal
            if node.untried_moves and not is_terminal_node(node.state): # Check if node state itself is terminal
                move = random.choice(node.untried_moves) # Expand randomly among untried
                current_player = node.player_at_node
                next_player = self.opponent_id if current_player == self.player_id else self.player_id

                # Apply the expansion move to the state inherited from selection
                row = get_next_open_row(current_board_state, move)
                if row is not None:
                    drop_piece(current_board_state, row, move, current_player)
                    node = node.add_child(move, current_board_state.copy(), next_player) # Add child with the *new* state
                else:
                    # This should not happen if untried_moves only contains valid moves
                    print(f"Warning: Attempted to expand invalid move {move}. Removing.")
                    node.untried_moves.remove(move)
                    continue # Skip to next iteration if expansion failed

            # 3. Simulation (Playout) - WITH HEURISTICS
            simulation_board = current_board_state.copy()
            # Player whose turn it is IN THE SIMULATION (starts from the 'node' state)
            sim_player = node.player_at_node

            # Check if the node state itself is terminal before starting simulation loop
            is_sim_terminal = is_terminal_node(simulation_board)

            while not is_sim_terminal:
                valid_moves = get_valid_locations(simulation_board)
                if not valid_moves:
                    is_sim_terminal = True # Draw
                    break

                # --- START: Heuristic Playout Move Selection ---
                # First priority: Look for winning moves
                chosen_move = None
                winning_move_found = False
                for m in valid_moves:
                    temp_board_win = simulation_board.copy()
                    r_win = get_next_open_row(temp_board_win, m)
                    drop_piece(temp_board_win, r_win, m, sim_player)
                    if winning_move(temp_board_win, sim_player):
                        chosen_move = m
                        winning_move_found = True
                        break
                if winning_move_found: pass
                else:
                    # Second priority: Look for blocking moves
                    sim_opponent = self.opponent_id if sim_player == self.player_id else self.player_id
                    blocking_move_found = False
                    for m in valid_moves:
                        temp_board_block = simulation_board.copy()
                        r_block = get_next_open_row(temp_board_block, m)
                        drop_piece(temp_board_block, r_block, m, sim_opponent)
                        if winning_move(temp_board_block, sim_opponent):
                             chosen_move = m
                             blocking_move_found = True
                             break
                    if blocking_move_found: pass
                # Third priority: Use positional values as a heuristic
                if not winning_move_found and not blocking_move_found:
                    move_values = {}
                    for m in valid_moves:
                        r = get_next_open_row(simulation_board, m)
                        move_values[m] = self.positional_values[r][m]
                    total_value = sum(v for v in move_values.values() if v > 0)
                    if total_value > 0:
                         # Choose moves with probability proportional to their positional value
                         weights = [max(0, move_values[m]) for m in valid_moves]
                         sum_weights = sum(weights)
                         if sum_weights > 0:
                             probabilities = [w / sum_weights for w in weights]
                             chosen_move = random.choices(valid_moves, weights=probabilities, k=1)[0]
                         else: chosen_move = random.choice(valid_moves)
                    else: chosen_move = random.choice(valid_moves)


                # Apply the chosen move
                row = get_next_open_row(simulation_board, chosen_move)
                drop_piece(simulation_board, row, chosen_move, sim_player)
                sim_player = self.opponent_id if sim_player == self.player_id else self.player_id # Switch player

                is_sim_terminal = is_terminal_node(simulation_board)


            # Determine simulation result (from the final simulation_board state)
            winner = None
            sim_draw = False
            last_player = self.opponent_id if sim_player == self.player_id else self.player_id # Player who made the last move
            if winning_move(simulation_board, last_player):
                 winner = last_player
            # Check for draw only if no winner
            elif not get_valid_locations(simulation_board) and not winner:
                 sim_draw = True

            sim_result_for_mcts_player = 0 # Draw is 0
            if winner == self.player_id:
                sim_result_for_mcts_player = 1  # Win is +1
            elif winner == self.opponent_id:
                sim_result_for_mcts_player = -1  # Loss is -1

            # 4. Backpropagation - update scores up the tree
            temp_node = node # Start backprop from the node where simulation started
            while temp_node is not None:
                 # Convert result to perspective of player at this node
                 result_for_node = sim_result_for_mcts_player if temp_node.player_at_node == self.player_id else -sim_result_for_mcts_player
                 temp_node.update(result_for_node)
                 temp_node = temp_node.parent

        # --- Choose the best move ---
        if not root.children:
             print(f"MCTS Player {self.player_id}: Warning - No moves explored/possible after {self.n_iterations} iterations. Choosing random.")
             return random.choice(get_valid_locations(board)) if get_valid_locations(board) else None # Fallback

        # Select child with highest number of visits (most robust)
        best_child = max(root.children, key=lambda c: c.visits)
        best_move = best_child.move

        # --- Display Info ---
        end_time = time.time()

        # Calculate win rate estimate for display
        parent_win_rate_for_best_child = (-best_child.score / best_child.visits) if best_child.visits > 0 else 0.0
        win_rate_display = (parent_win_rate_for_best_child + 1) / 2 * 100  # Convert to percentage

        print(f"MCTS Player {self.player_id}: Chose column {best_move} "
              f"({best_child.visits} visits, "
              f"~WinRate: {win_rate_display:.1f}%, "
              f"Time: {end_time - start_time:.2f}s)")

        # Final validation check
        if best_move not in get_valid_locations(board):
             print(f"MCTS Warning: Chosen best move {best_move} is invalid! Fallback to random.")
             valid_fallback = get_valid_locations(board)
             return random.choice(valid_fallback) if valid_fallback else None

        return best_move
    
    
    
############################################################################
################# Q-agent ##################################################
############################################################################
    
    


class QLearningAgent(Player):
    """ A Reinforcement Learning agent using Q-learning for Connect 4. """
    def __init__(self, player_id, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, exploration_decay=0.9995, min_exploration_rate=0.01):
        super().__init__(player_id)
        self.opponent_id = PLAYER1_PIECE if player_id == PLAYER2_PIECE else PLAYER2_PIECE
        self.q_table = {}  # Dictionary to store state-action values
        self.lr = learning_rate  # Learning rate (alpha)
        self.gamma = discount_factor  # Discount factor for future rewards
        self.epsilon = exploration_rate  # Exploration rate (initially high)
        self.epsilon_decay = exploration_decay  # Rate at which exploration decreases
        self.min_epsilon = min_exploration_rate  # Minimum exploration rate
        self.previous_state_tuple = None  # Previous state for learning updates
        self.previous_action = None  # Previous action for learning updates
        self.is_learning = True  # Whether the agent should learn or just use its policy
        self.verbose = False  # Whether to print detailed information

    def _state_to_tuple(self, board):
        # Convert board (numpy array) to immutable tuple format for Q-table dictionary keys
        return tuple(map(tuple, board))
    
    def _flip_state(self, board):
        """Flips board perspective so Player 2 can use P1's learned values"""
        # Create a deep copy of the board to avoid modifying the original
        flipped_board = board.copy()
        p1_mask, p2_mask = flipped_board == PLAYER1_PIECE, flipped_board == PLAYER2_PIECE
        flipped_board[p1_mask], flipped_board[p2_mask] = PLAYER2_PIECE, PLAYER1_PIECE
        return flipped_board
   
    def get_q_value(self, state_tuple, action):
        # Retrieve Q-value for a state-action pair, return 0.0 if not found in Q-table
        return self.q_table.get((state_tuple, action), 0.0)
   
    def choose_action(self, board):
        """Core logic to select an action using epsilon-greedy policy"""
        # For P2, flip board so it can use P1's learned values
        if self.player_id == PLAYER1_PIECE: lookup_board = board
        else: lookup_board = self._flip_state(board)
        # Convert board to tuple for Q-table lookup
        state_tuple = self._state_to_tuple(lookup_board)
        valid_actions = get_valid_locations(board)
        # Initialize details dictionary for debugging and tracking decision process
        details = {'max_q': None, 'chosen_q': None, 'tie_break_used': False, 'exploited': False}
        if not valid_actions: return None, details
        
        # Exploration: choose random action with probability epsilon (exploration strategy)
        if self.is_learning and random.uniform(0, 1) < self.epsilon:
            # Random exploration move
            chosen_action = random.choice(valid_actions)
            details['chosen_q'] = self.get_q_value(state_tuple, chosen_action); details['exploited'] = False
            return chosen_action, details
        else:
            # Exploitation: choose action with highest Q-value (use learned knowledge)
            details['exploited'] = True
            # Create dictionary of Q-values for all valid actions
            q_values = {action: self.get_q_value(state_tuple, action) for action in valid_actions}
            if not q_values: return random.choice(valid_actions), details
            
            # Find maximum Q-value among all valid actions
            max_q = -float('inf'); tolerance = 1e-9
            for q in q_values.values():
                 if q > max_q: max_q = q
            details['max_q'] = max_q
            
            # Find all actions with Q-values close to maximum (within tolerance)
            best_actions = [action for action, q in q_values.items() if abs(q - max_q) < tolerance]
            
            chosen_action = None
            if len(best_actions) == 1: chosen_action = best_actions[0]  # Single best action - clear choice
            elif len(best_actions) > 1:
                # Tie-breaking using position heuristics when multiple actions have same Q-value
                details['tie_break_used'] = True; best_heuristic_value = -float('inf'); tied_heuristic_actions = []
                for action in best_actions:
                    row = get_next_open_row(board, action)
                    if row is not None:
                         # Use positional value as heuristic for tie-breaking
                         heuristic_value = POSITIONAL_VALUES[row][action]
                         if heuristic_value > best_heuristic_value: best_heuristic_value = heuristic_value; tied_heuristic_actions = [action]
                         elif heuristic_value == best_heuristic_value: tied_heuristic_actions.append(action)
                # Choose randomly among positions with best heuristic value
                if tied_heuristic_actions: chosen_action = random.choice(tied_heuristic_actions)
                else: chosen_action = random.choice(best_actions)
            else: chosen_action = random.choice(valid_actions)  # Fallback if no best actions found
            
            details['chosen_q'] = q_values.get(chosen_action, 0.0)
            return chosen_action, details


    def get_move(self, board):
        """Gets the agent's move. Prints details ONLY if self.verbose is True."""
        start_time = time.time()
        # Get chosen action and details from the decision making process
        chosen_action, details = self.choose_action(board.copy())
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Handle case where no valid moves are available
        if chosen_action is None:
            print(f"Q-Agent {self.player_id}: No valid moves available.")
            return None

        # Print verbose information if enabled - for debugging and monitoring agent decisions
        if self.verbose:
            max_q_str = f"{details['max_q']:.4f}" if details['max_q'] is not None else "N/A"
            chosen_q_str = f"{details['chosen_q']:.4f}" if details['chosen_q'] is not None else "N/A"
            mode = "Exploit" if details['exploited'] else "Explore"
            tie_info = "(TIEBREAK)" if details['tie_break_used'] else ""

            print(f"Q-Agent {self.player_id}: Chose column {chosen_action} "
                  f"({mode}{tie_info}, "
                  f"ChosenQ: {chosen_q_str}, MaxQ: {max_q_str}, "
                  f"Time: {elapsed_time:.3f}s)")

        # Store current state and action for learning in the next step
        if self.is_learning:
            self.previous_state_tuple = self._state_to_tuple(board)
            self.previous_action = chosen_action

        return chosen_action


    def learn(self, reward, next_board):
        # Skip learning if not in learning mode or if no previous state/action exists
        if not self.is_learning or self.previous_state_tuple is None or self.previous_action is None: return
        old_q = self.get_q_value(self.previous_state_tuple, self.previous_action)
        # For P2, flip board to use P1's perspective for consistent learning
        if self.player_id == PLAYER1_PIECE: lookup_next_board = next_board
        else: lookup_next_board = self._flip_state(next_board)
        
        next_state_tuple = self._state_to_tuple(lookup_next_board)
        valid_next_actions = get_valid_locations(next_board)
        max_future_q = 0.0
        
        # If not terminal state and valid actions exist, find max Q-value for next state
        if not is_terminal_node(next_board) and valid_next_actions:
            q_values_next = [self.get_q_value(next_state_tuple, action) for action in valid_next_actions]
            if q_values_next: max_future_q = max(q_values_next)
        
        # Apply Q-learning update formula: Q(s,a) = Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        new_q = old_q + self.lr * (reward + self.gamma * max_future_q - old_q)
        self.q_table[(self.previous_state_tuple, self.previous_action)] = new_q

    def update_epsilon(self):
        if self.is_learning: self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filepath="q_table.pkl"):
        # Use atomic write pattern with temporary file to prevent corruption
        temp_filepath = filepath + ".tmp"; final_filepath = filepath
        try:
            with open(temp_filepath, 'wb') as f: pickle.dump(self.q_table, f)
            os.replace(temp_filepath, final_filepath)
            print(f"Q-table saved successfully to {final_filepath} ({len(self.q_table)} entries)")
        except Exception as e:
            print(f"Error saving Q-table: {e}")
            if os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except Exception as e_rem: print(f"Error removing temp file: {e_rem}")

    def load_q_table(self, filepath="q_table.pkl"):
        try:
            # Check if Q-table file exists
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f: self.q_table = pickle.load(f)
                print(f"Q-table loaded successfully from {filepath} ({len(self.q_table)} entries)")
            else:
                # Start with empty Q-table if file not found
                print(f"Q-table file not found at {filepath}. Starting fresh."); self.q_table = {}
        except (pickle.UnpicklingError, EOFError, ValueError) as e_load:
            print(f"Error loading Q-table from {filepath}: File might be corrupted or incompatible. {e_load}")
            print("Starting with an empty Q-table."); self.q_table = {}
        except Exception as e:
            print(f"An unexpected error occurred loading Q-table: {e}"); self.q_table = {}

    
    
    
############################################################################
################# Hybrid MCTS Q-agent ######################################
############################################################################
    

class MCTS_QAgent_Hybrid(Player):
    """ Combines MCTS search with Q-Agent guided simulations. """

    def __init__(self, player_id, q_table_path, iterations=1000, exploration_constant=1.414):
        super().__init__(player_id)
        self.opponent_id = PLAYER1_PIECE if player_id == PLAYER2_PIECE else PLAYER2_PIECE
        self.n_iterations = iterations
        self.exploration_constant = exploration_constant

        self.q_agent = None
        print(f"Initializing Hybrid Agent (Player {player_id})")
        try:
            # Load a pre-trained Q-learning agent with exploration disabled
            self.q_agent = QLearningAgent(player_id=PLAYER1_PIECE)
            self.q_agent.load_q_table(q_table_path)
            if not self.q_agent.q_table:
                print(f"WARNING: Q-table file '{q_table_path}' loaded empty for Hybrid agent.")
            self.q_agent.is_learning = False  # We don't want to learn during gameplay
            self.q_agent.epsilon = 0.0  # No exploration, just use learned values
            self.q_agent.verbose = False
            print(f"Internal Q-Agent loaded ({len(self.q_agent.q_table)} entries) for Hybrid Player {player_id}.")
        except NameError:
            print("ERROR: QLearningAgent class not defined. Cannot create Hybrid Agent.")
            raise
        except FileNotFoundError:
             print(f"ERROR: Q-table file '{q_table_path}' not found for Hybrid Agent.")
             raise
        except Exception as e:
            print(f"ERROR loading Q-table for Hybrid Agent: {e}")
            raise

        if self.q_agent is None:
             raise ValueError("Failed to initialize internal Q-Agent for Hybrid player.")


    def _simulate_with_q_agent(self, board_state, starting_player):
        """ Performs a playout using the Q-agent to choose moves. """
        simulation_board = board_state.copy()
        sim_player = starting_player

        try: 
            initial_terminal_check = is_terminal_node(simulation_board)
            is_sim_terminal = initial_terminal_check

            while not is_sim_terminal:
                valid_moves = get_valid_locations(simulation_board)
                if not valid_moves:
                    is_sim_terminal = True; break

                # The Q-agent is trained as player 1, so we need to flip the board
                # when making decisions as player 2
                board_for_q_eval = None
                if sim_player == PLAYER1_PIECE:
                    board_for_q_eval = simulation_board
                else:
                    board_for_q_eval = self.q_agent._flip_state(simulation_board)

                # Let the Q-agent choose the next move based on its learned values
                chosen_move, _ = self.q_agent.choose_action(board_for_q_eval)

                # Fallback to random move if Q-agent returns invalid move
                if chosen_move is None or chosen_move not in valid_moves:
                    chosen_move = random.choice(valid_moves)

                row = get_next_open_row(simulation_board, chosen_move)
                if row is None:
                     print(f"Warning: Q-Sim chose invalid move {chosen_move} - row is None.")
                     is_sim_terminal = True; break

                # Make the move and switch players
                drop_piece(simulation_board, row, chosen_move, sim_player)
                sim_player = PLAYER2_PIECE if sim_player == PLAYER1_PIECE else PLAYER1_PIECE

                loop_terminal_check = is_terminal_node(simulation_board)
                is_sim_terminal = loop_terminal_check

        except Exception as sim_e:
             print(f"\n!!! ERROR during simulation !!!")
             print(f"Player whose turn it was: {sim_player}")
             print(f"Board state where error occurred:\n{simulation_board}")
             print(f"Error details: {type(sim_e)} - {sim_e}")
             return -1  # Return loss in case of error

        winner = None
        last_player = PLAYER2_PIECE if sim_player == PLAYER1_PIECE else PLAYER1_PIECE

        try:
             final_win_check_result = winning_move(simulation_board, last_player)
        except Exception as win_e:
             print(f"\n!!! ERROR during final win check !!!")
             print(f"Board state:\n{simulation_board}")
             print(f"Error details: {type(win_e)} - {win_e}")
             final_win_check_result = False 

        if final_win_check_result:
            winner = last_player
        elif not get_valid_locations(simulation_board):
            pass  # Draw if no valid moves left

        # Convert result to agent's perspective: win=1, loss=-1, draw=0
        sim_result = 0  # Default to draw
        if winner == self.player_id: sim_result = 1
        elif winner == self.opponent_id: sim_result = -1
        return sim_result


    def get_move(self, board):
        """ Determines the move using MCTS guided by Q-Agent simulations. """
        start_time = time.time()
        root = MCTSNode(state=board.copy(), player_at_node=self.player_id)

        valid_locations = get_valid_locations(board)

        # Check for immediate win
        for col in valid_locations:
             temp_board_win = board.copy()
             row = get_next_open_row(temp_board_win, col)
             if row is not None:
                 drop_piece(temp_board_win, row, col, self.player_id)
                 win_check_result = winning_move(temp_board_win, self.player_id)
                 if win_check_result:
                     print(f"Hybrid Player {self.player_id}: Found immediate winning move {col}")
                     return col

        # Block opponent's immediate win
        for col in valid_locations:
             temp_board_loss = board.copy()
             row = get_next_open_row(temp_board_loss, col)
             if row is not None:
                 drop_piece(temp_board_loss, row, col, self.opponent_id)
                 block_check_result = winning_move(temp_board_loss, self.opponent_id)
                 if block_check_result:
                     print(f"Hybrid Player {self.player_id}: Found immediate block at {col}")
                     return col

        # Main MCTS loop - Run for specified number of iterations
        for i in range(self.n_iterations):
            node = root
            current_board_state = board.copy()

            # 1. Selection - Follow UCT down the tree until leaf node
            while not node.untried_moves and node.children:
                node = node.uct_select_child(self.exploration_constant)
                if node is None: break
                row = get_next_open_row(current_board_state, node.move)
                if row is None: break
                drop_piece(current_board_state, row, node.move, node.parent.player_at_node)
            if node is None: continue

            # 2. Expansion - Add a new child node if not terminal
            try:
                node_is_terminal = is_terminal_node(node.state)
            except Exception as e_debug:
                print(f"DEBUG (Hybrid P{self.player_id}): Iter {i}, ERROR checking node terminal: {e_debug}")
                node_is_terminal = True 


            if node.untried_moves and not node_is_terminal:
                move = random.choice(node.untried_moves)
                current_player = node.player_at_node
                next_player = self.opponent_id if current_player == self.player_id else self.player_id
                row = get_next_open_row(current_board_state, move)
                if row is not None:
                    board_after_expansion = current_board_state.copy() 
                    drop_piece(board_after_expansion, row, move, current_player)
                    node = node.add_child(move, board_after_expansion, next_player) 
                    current_board_state = board_after_expansion
                else:
                    node.untried_moves.remove(move); continue

            # 3. Simulation (using Q-Agent instead of random rollout)
            simulation_result = self._simulate_with_q_agent(current_board_state, node.player_at_node)


            # 4. Backpropagation - Update all nodes from selected node to root
            temp_node = node
            while temp_node is not None:
                 result_for_node = simulation_result if temp_node.player_at_node == self.player_id else -simulation_result
                 temp_node.update(result_for_node)
                 temp_node = temp_node.parent


        # --- Choose Best Move Based on Visit Count ---
        if not root.children:
             print(f"Hybrid Player {self.player_id}: Warning - No moves explored. Choosing random.")
             valid_fallback = get_valid_locations(board)
             return random.choice(valid_fallback) if valid_fallback else None

        # Select the move with the most visits (most robust selection strategy)
        best_child = max(root.children, key=lambda c: c.visits)
        best_move = best_child.move
        end_time = time.time()

        # Calculate and display win rate for logging purposes
        parent_win_rate_for_best_child = (-best_child.score / best_child.visits) if best_child.visits > 0 else 0.0
        win_rate_display = (parent_win_rate_for_best_child + 1) / 2 * 100

        print(f"Hybrid Player {self.player_id}: Chose column {best_move} "
              f"({best_child.visits} visits, ~WinRate: {win_rate_display:.1f}%, "
              f"Time: {end_time - start_time:.2f}s)")

        # Final safety check - ensure the chosen move is valid
        if best_move not in get_valid_locations(board):
             print(f"Hybrid Warning: Chosen best move {best_move} is invalid! Fallback.")
             valid_fallback = get_valid_locations(board)
             return random.choice(valid_fallback) if valid_fallback else None

        return best_move
    

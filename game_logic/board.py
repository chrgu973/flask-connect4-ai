import numpy as np
import copy
import time

# --- Constants ---
ROWS = 6  # Standard Connect 4 has 6 rows
COLS = 7  # Standard Connect 4 has 7 columns
EMPTY = 0  # Empty cell value
PLAYER1_PIECE = 1  # Player 1 piece (Yellow)
PLAYER2_PIECE = 2  # Player 2 piece (Red)


# Positional Heuristic Values (Higher value = better center/strategic position)
# Defined visually top-down for clarity - center positions are more valuable
POSITIONAL_VALUES_RAW = np.array([
    [3, 4, 5, 7, 5, 4, 3],  # Top row (least valuable)
    [4, 6, 8, 10, 8, 6, 4],
    [5, 8, 11, 13, 11, 8, 5],
    [5, 8, 11, 13, 11, 8, 5],
    [4, 6, 8, 10, 8, 6, 4],
    [3, 4, 5, 7, 5, 4, 3]   # Bottom row
])

# Flip the board to match the internal representation (row 0 is bottom)
POSITIONAL_VALUES = np.flipud(POSITIONAL_VALUES_RAW)



# --- Board Functions ---
def create_board():
    """Creates an empty Connect 4 board."""
    return np.zeros((ROWS, COLS), dtype=int)

def print_board(board):
    """Prints the board to the console in a formatted way."""
    # Map internal values to display characters
    piece_map = {
        EMPTY: " ",
        PLAYER1_PIECE: "X", # Player 1
        PLAYER2_PIECE: "O"  # Player 2
    }

    # Print the board rows from top to bottom (needs flipping)
    flipped_board = np.flip(board, 0)
    for r in range(ROWS):
        row_str = "| " # Start of the row border
        # Join pieces with spaces in between
        row_str += " ".join([piece_map[flipped_board[r][c]] for c in range(COLS)])
        row_str += " |" # End of the row border
        print(row_str)

    # Print the bottom border
    print("+" + "-" * (COLS * 2 + 1) + "+") # Adjust width based on spacing

    # Print the column numbers (1-7) aligned below
    col_numbers = "  " + " ".join(map(str, range(1, COLS + 1)))
    print(col_numbers)

def is_valid_location(board, col):
    """Checks if a column is valid for dropping a piece."""
    return 0 <= col < COLS and board[ROWS - 1][col] == EMPTY 

def get_next_open_row(board, col):
    """Finds the lowest empty row in a given column."""
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            return r
    return None 

def drop_piece(board, row, col, piece):
    """Places a piece on the board at the specified location."""
    board[row][col] = piece

def get_valid_locations(board):
    """Returns a list of columns where a piece can be dropped."""
    return [col for col in range(COLS) if is_valid_location(board, col)]


def winning_move(board, piece):
    """Checks if the specified player has won."""
    # Check horizontal locations
    for c in range(COLS - 3):
        for r in range(ROWS):
            if all(board[r][c+i] == piece for i in range(4)):
                return True

    # Check vertical locations
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r+i][c] == piece for i in range(4)):
                return True

    # Check positively sloped diagonals
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return True

    # Check negatively sloped diagonals
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if all(board[r-i][c+i] == piece for i in range(4)):
                return True

    return False

def is_terminal_node(board):
    """Checks if the game has ended (win or draw)."""
    return winning_move(board, PLAYER1_PIECE) or \
           winning_move(board, PLAYER2_PIECE) or \
           len(get_valid_locations(board)) == 0


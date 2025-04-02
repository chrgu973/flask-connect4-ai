# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import numpy as np
import time
import os

# Import your game logic and player classes
from game_logic.board import (
    create_board, is_valid_location, get_next_open_row,
    drop_piece, winning_move, is_terminal_node, get_valid_locations,
    PLAYER1_PIECE, PLAYER2_PIECE, ROWS, COLS, EMPTY
)
from game_logic.players import (
    Player, HumanPlayer, MCTS_QAgent_Hybrid
)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Global AI Initialization (Instance) ---
AI_PLAYER = None
Q_TABLE_PATH = "connect4_q_agent_selfplay_gen2.pkl"
HYBRID_ITERATIONS = 1500

try:
    print("Initializing Hybrid AI Player Instance...")
    AI_PLAYER = MCTS_QAgent_Hybrid(
        player_id=PLAYER1_PIECE, # Default, overridden by session logic
        q_table_path=Q_TABLE_PATH,
        iterations=HYBRID_ITERATIONS
    )
    print("Hybrid AI Player Instance Initialized Successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not initialize AI Player Instance: {e}")
    AI_PLAYER = None

# --- Constants for this app's color convention ---
PLAYER1_COLOR = "Yellow"
PLAYER2_COLOR = "Red"

# --- Session Helper Functions (Keep as before) ---
def get_game_state():
    # ... (existing code) ...
    if 'board' not in session or 'human_player_id' not in session:
        return {'game_setup_complete': False}
    return {
        'game_setup_complete': True, 'board': session.get('board'), 'turn': session.get('turn'),
        'human_player_id': session.get('human_player_id'), 'ai_player_id': session.get('ai_player_id'),
        'winner': session.get('winner', None), 'message': session.get('message', ""),
        'game_active': session.get('game_active', False), 'needs_ai_first_move': session.get('needs_ai_first_move', False)
    }

def update_game_state(**kwargs):
    # ... (existing code) ...
    state_changed = False
    for key, value in kwargs.items():
        if session.get(key) != value:
            session[key] = value
            state_changed = True
    if state_changed:
        session.modified = True

# --- Routes ---

@app.route('/')
def index():
    # ... (existing code - renders choose_player.html or index.html) ...
    state = get_game_state()
    if not state['game_setup_complete']:
        return render_template('choose_player.html')
    else:
        is_player_turn_now = (state['turn'] == state['human_player_id'] and state['game_active'])
        return render_template('index.html',
                               board=np.array(state['board']), rows=ROWS, cols=COLS,
                               message=state['message'], game_active=state['game_active'],
                               is_player_turn=is_player_turn_now,
                               human_player_id=state['human_player_id'], # Pass to JS
                               # ai_player_id=state['ai_player_id'], # Less critical for JS now
                               needs_ai_first_move=state['needs_ai_first_move'],
                               board_json=jsonify(state['board']).get_data(as_text=True)
                               )


@app.route('/start')
def start_game():
    """Starts a new game based on player choice. P1=Yellow, P2=Red."""
    if AI_PLAYER is None:
         flash("Error: AI opponent could not be loaded. Cannot start game.", "error")
         return redirect(url_for('index'))

    choice = request.args.get('player', type=int)

    if choice == 1: # Human chose P1 (Yellow, First)
        human_id = PLAYER1_PIECE
        ai_id = PLAYER2_PIECE
        initial_turn = human_id
        needs_ai_move = False
        message = f"Your turn ({PLAYER1_COLOR})" # <-- UPDATED MESSAGE
    elif choice == 2: # Human chose P2 (Red, Second)
        human_id = PLAYER2_PIECE
        ai_id = PLAYER1_PIECE
        initial_turn = ai_id
        needs_ai_move = True
        message = f"Starting Game... AI ({PLAYER1_COLOR}) is thinking..." # <-- UPDATED MESSAGE
    else:
        flash("Invalid player choice.", "error")
        return redirect(url_for('index'))

    clear_game_session()
    board_np = create_board()
    update_game_state(
        board=board_np.tolist(), human_player_id=human_id, ai_player_id=ai_id,
        turn=initial_turn, winner=None, message=message, game_active=True,
        needs_ai_first_move=needs_ai_move
    )
    print(f"New game started. Human: P{human_id} ({PLAYER2_COLOR if human_id==2 else PLAYER1_COLOR}), AI: P{ai_id} ({PLAYER1_COLOR if ai_id==1 else PLAYER2_COLOR}), Turn: P{initial_turn}")
    return redirect(url_for('index'))


@app.route('/ai_first_move', methods=['POST'])
def ai_first_move():
    """Handles the AI's first move (always P1=Yellow in this scenario)."""
    state = get_game_state()
    if AI_PLAYER is None or not state.get('game_setup_complete'):
        return jsonify({'error': 'AI Player not initialized or game not set up'}), 500

    ai_id = state.get('ai_player_id')
    human_id = state.get('human_player_id')
    # Sanity check: AI first move only happens if AI is P1
    if ai_id != PLAYER1_PIECE:
        return jsonify({'error': 'AI first move called but AI is not Player 1'}), 400

    human_color = PLAYER2_COLOR # Human is P2 in this case

    if not state.get('needs_ai_first_move') or state.get('turn') != ai_id:
        print("AI first move already made or not AI's turn.")
        return jsonify({
            'board': state['board'], 'message': state['message'], 'game_active': state['game_active'],
            'is_player_turn': (state['turn'] == human_id and state['game_active'])
        })

    print(f"Processing AI (P{ai_id}={PLAYER1_COLOR})'s first move...")
    board_np = create_board()

    try:
        ai_start_time = time.time()
        AI_PLAYER.player_id = ai_id
        AI_PLAYER.opponent_id = human_id
        col = AI_PLAYER.get_move(board_np.copy())
        ai_end_time = time.time()
        print(f"AI (P{ai_id}) initial move: {col} (took {ai_end_time - ai_start_time:.2f}s)")

        if col is not None and is_valid_location(board_np, col):
            row = get_next_open_row(board_np, col)
            drop_piece(board_np, row, col, ai_id)
            message = f"Your turn ({human_color})" # <-- UPDATED MESSAGE
            turn = human_id
            update_game_state(board=board_np.tolist(), turn=turn, message=message, needs_ai_first_move=False)
            return jsonify({
                'board': board_np.tolist(), 'message': message, 'game_active': True, 'is_player_turn': True
            })
        else:
            # ... (error handling as before) ...
            print(f"ERROR: AI (P{ai_id}) made invalid first move {col}")
            message = "Error: AI failed first move. Try New Game."
            update_game_state(message=message, game_active=False, needs_ai_first_move=False)
            return jsonify({'error': message}), 500
    except Exception as e:
        # ... (error handling as before) ...
        print(f"Error during AI (P{ai_id})'s first move: {e}")
        message = "Error during AI's first move. Try New Game."
        update_game_state(message=message, game_active=False, needs_ai_first_move=False)
        return jsonify({'error': message}), 500


@app.route('/play', methods=['POST'])
def play():
    """Handles the Human's move."""
    state = get_game_state()
    # ... (initial checks as before) ...
    if AI_PLAYER is None or not state.get('game_setup_complete'): return jsonify({'error': 'AI Player not initialized or game not set up'}), 500
    board_np = np.array(state['board'])
    current_turn = state['turn']
    game_active = state['game_active']
    human_id = state['human_player_id']
    ai_id = state['ai_player_id']
    if not game_active: return jsonify({'error': 'Game is not active'}), 400
    if current_turn != human_id: return jsonify({'error': 'Not your turn'}), 400

    data = request.get_json()
    col = data.get('column')

    # Determine colors based on standard P1=Yellow, P2=Red convention
    human_color = PLAYER1_COLOR if human_id == PLAYER1_PIECE else PLAYER2_COLOR
    ai_color = PLAYER1_COLOR if ai_id == PLAYER1_PIECE else PLAYER2_COLOR

    if col is None or not isinstance(col, int) or not is_valid_location(board_np, col):
        # ... (invalid move handling as before, using human_color) ...
        print(f"Human (P{human_id}) tried invalid column {col}")
        message = f"Invalid move (Column {col+1}?). Your turn ({human_color})." # <-- UPDATED MESSAGE
        return jsonify({ 'board': board_np.tolist(), 'message': message, 'game_active': True, 'is_player_turn': True, 'trigger_ai': False }), 400

    # --- Process Valid Human Move ---
    row = get_next_open_row(board_np, col)
    drop_piece(board_np, row, col, human_id)
    print(f"Human (P{human_id}) placed piece in column {col}")

    # Check for human win
    if winning_move(board_np, human_id):
        message = f"Congratulations! You win! ({human_color})" # <-- UPDATED MESSAGE
        update_game_state(board=board_np.tolist(), winner=human_id, message=message, game_active=False, turn=None)
        return jsonify({ 'board': board_np.tolist(), 'message': message, 'game_active': False, 'is_player_turn': False, 'trigger_ai': False })

    # Check for draw
    if len(get_valid_locations(board_np)) == 0:
        # ... (draw handling as before) ...
         message = "It's a DRAW!"
         update_game_state(board=board_np.tolist(), winner=EMPTY, message=message, game_active=False, turn=None)
         return jsonify({ 'board': board_np.tolist(), 'message': message, 'game_active': False, 'is_player_turn': False, 'trigger_ai': False })


    # --- Human move valid, game continues ---
    message = f"AI's turn ({ai_color})... Thinking..." # <-- UPDATED MESSAGE
    update_game_state(board=board_np.tolist(), turn=ai_id, message=message)

    return jsonify({
        'board': board_np.tolist(), 'message': message, 'game_active': True,
        'is_player_turn': False, 'trigger_ai': True
    })


@app.route('/ai_move', methods=['POST'])
def ai_move():
    """Handles the AI's move processing."""
    state = get_game_state()
    # ... (initial checks as before) ...
    if AI_PLAYER is None or not state.get('game_setup_complete'): return jsonify({'error': 'AI Player not initialized or game not set up'}), 500
    board_np = np.array(state['board'])
    current_turn = state['turn']
    game_active = state['game_active']
    ai_id = state['ai_player_id']
    human_id = state['human_player_id']
    if not game_active: return jsonify({'error': 'Game is not active'}), 400
    if current_turn != ai_id: return jsonify({'error': 'Not AI turn in state'}), 400

    # Determine colors based on standard P1=Yellow, P2=Red convention
    human_color = PLAYER1_COLOR if human_id == PLAYER1_PIECE else PLAYER2_COLOR
    ai_color = PLAYER1_COLOR if ai_id == PLAYER1_PIECE else PLAYER2_COLOR

    print(f"Processing AI (P{ai_id}={ai_color}) move...")

    try:
        ai_start_time = time.time()
        AI_PLAYER.player_id = ai_id
        AI_PLAYER.opponent_id = human_id
        ai_col = AI_PLAYER.get_move(board_np.copy())
        ai_end_time = time.time()
        print(f"AI (P{ai_id}) response move: {ai_col} (took {ai_end_time - ai_start_time:.2f}s)")

        if ai_col is not None and is_valid_location(board_np, ai_col):
            ai_row = get_next_open_row(board_np, ai_col)
            drop_piece(board_np, ai_row, ai_col, ai_id)

            # Check for AI win
            if winning_move(board_np, ai_id):
                message = f"AI Wins! ({ai_color})" # <-- UPDATED MESSAGE
                update_game_state(board=board_np.tolist(), winner=ai_id, message=message, game_active=False, turn=None)
                return jsonify({ 'board': board_np.tolist(), 'message': message, 'game_active': False, 'is_player_turn': False })

            # Check for draw
            if len(get_valid_locations(board_np)) == 0:
                # ... (draw handling as before) ...
                 message = "It's a DRAW!"
                 update_game_state(board=board_np.tolist(), winner=EMPTY, message=message, game_active=False, turn=None)
                 return jsonify({ 'board': board_np.tolist(), 'message': message, 'game_active': False, 'is_player_turn': False })

            # Game continues, it's human's turn again
            message = f"Your turn ({human_color})" # <-- UPDATED MESSAGE
            update_game_state(board=board_np.tolist(), turn=human_id, message=message)
            return jsonify({
                'board': board_np.tolist(), 'message': message, 'game_active': True, 'is_player_turn': True
            })

        else: # AI returned invalid move
             # ... (error handling as before, using human_color) ...
             print(f"ERROR: AI (P{ai_id}) chose invalid column {ai_col}")
             message = f"Error: AI made invalid move. Your turn ({human_color})." # <-- UPDATED MESSAGE
             update_game_state(turn=human_id, message=message)
             return jsonify({ 'board': board_np.tolist(), 'message': message, 'game_active': True, 'is_player_turn': True })

    except Exception as e:
         # ... (error handling as before, using human_color) ...
         print(f"Error during AI (P{ai_id})'s move processing: {e}")
         message = f"Error during AI move. Your turn ({human_color})." # <-- UPDATED MESSAGE
         update_game_state(turn=human_id, message=message)
         return jsonify({ 'board': board_np.tolist(), 'message': message, 'game_active': True, 'is_player_turn': True }), 500


# --- clear_game_session and reset routes (Keep as before) ---
def clear_game_session():
    # ... (existing code) ...
    keys_to_clear = ['board', 'turn', 'winner', 'message', 'game_active', 'needs_ai_first_move', 'human_player_id', 'ai_player_id']
    for key in keys_to_clear: session.pop(key, None)
    session.modified = True
    print("Cleared game session keys.")

@app.route('/reset')
def reset():
    # ... (existing code) ...
    clear_game_session()
    flash("Game Reset. Choose your player.", "info")
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
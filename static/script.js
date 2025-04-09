// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const boardElement = document.getElementById('board');
    const messageArea = document.getElementById('message-area');
    const newGameBtn = document.getElementById('new-game-btn');
    const gameArea = document.getElementById('game-area');
    const cells = document.querySelectorAll('.cell');

    // --- Initialization ---
    const localInitialBoard = typeof initialBoard !== 'undefined' ? initialBoard : null;
    let localGameActive = typeof gameActive !== 'undefined' ? gameActive : false;
    let localIsPlayerTurn = typeof isPlayerTurn !== 'undefined' ? isPlayerTurn : false;
    const localNeedsAIFirstMove = typeof initialNeedsAIFirstMove !== 'undefined' ? initialNeedsAIFirstMove : false;
    const humanPlayerId = typeof localHumanPlayerId !== 'undefined' ? localHumanPlayerId : null;

    let currentBoard = localInitialBoard;

    // --- Functions ---
    function updateBoard(boardData) {
        currentBoard = boardData; 
        if (!currentBoard) return;

        let cellIndex = 0;
        for (let r = boardData.length - 1; r >= 0; r--) {
            for (let c = 0; c < boardData[0].length; c++) {
                if (cellIndex >= cells.length) break;
                const pieceDiv = cells[cellIndex].querySelector('.piece');
                if (!pieceDiv) continue;

                pieceDiv.classList.remove('player1-color', 'player2-color');

                const pieceValue = boardData[r][c];
                if (pieceValue === 1) { 
                    pieceDiv.classList.add('player1-color');
                } else if (pieceValue === 2) { 
                    pieceDiv.classList.add('player2-color');
                }
                cellIndex++;
            }
        }
    }


    function updateMessage(message) {
        if (messageArea) {
             messageArea.innerHTML = message;
        }
    }

    function setTurnIndicator() {
         if (!cells || cells.length === 0) return;
         cells.forEach(cell => {
            cell.classList.remove('player-turn-hover');
            if (localGameActive && localIsPlayerTurn) {
                const col = parseInt(cell.dataset.col);
                 if (currentBoard && currentBoard.length > 0 && currentBoard[0]) {
                     const topRowIndex = currentBoard.length - 1;
                     if (currentBoard[topRowIndex][col] === 0) {
                         for(let i=0; i < cells.length; i++){
                             if(parseInt(cells[i].dataset.col) === col){
                                 cells[i].classList.add('player-turn-hover');
                             }
                         }
                     }
                 }
             }
         });
     }

    // --- Request AI move --- 
    async function requestAIMove() {
        console.log("Requesting AI move from /ai_move");
        try {
            const response = await fetch('/ai_move', {
                method: 'POST',
                 headers: { 'Content-Type': 'application/json', },
            });
            if (!response.ok) {
                let errorMsg = `HTTP error! status: ${response.status}`;
                try { const errorData = await response.json(); errorMsg = errorData.error || errorMsg; } catch (e) { /* Ignore */ }
                throw new Error(errorMsg);
            }
            const data = await response.json();
            console.log("Received from /ai_move:", data);
            updateBoard(data.board);
            updateMessage(data.message);
            localGameActive = data.game_active;
            localIsPlayerTurn = data.is_player_turn;
            if (!localGameActive) { console.log("Game over after AI move."); }
            setTurnIndicator();
        } catch (error) {
            console.error('Error during AI move request:', error);
            updateMessage(`Error getting AI move: ${error.message}. Please refresh.`);
            localGameActive = false;
            localIsPlayerTurn = false;
            setTurnIndicator();
        }
    }

     // --- Request AI's FIRST move ---
     async function requestAIFirstMove() {
        console.log("Requesting AI first move from /ai_first_move");
        updateMessage("Starting Game... AI is thinking...");
        try {
            const response = await fetch('/ai_first_move', { method: 'POST', });
             if (!response.ok) {
                let errorMsg = `HTTP error! status: ${response.status}`;
                try { const errorData = await response.json(); errorMsg = errorData.error || errorMsg; } catch (e) {}
                throw new Error(errorMsg);
            }
            const data = await response.json();
            console.log("Received from /ai_first_move:", data);
            updateBoard(data.board);
            updateMessage(data.message);
            localGameActive = data.game_active;
            localIsPlayerTurn = data.is_player_turn;
            setTurnIndicator();
        } catch (error) {
            console.error('Error getting AI first move:', error);
            updateMessage(`Error starting game: ${error.message}. Please refresh.`);
            localGameActive = false;
            localIsPlayerTurn = false;
            setTurnIndicator();
        }
    }


    // --- Event Listeners ---
    if (boardElement) {
        boardElement.addEventListener('click', async (event) => {
            // Uses humanPlayerId to check if it's the human's turn
            if (!localGameActive || !localIsPlayerTurn || !humanPlayerId) {
                return;
            }
            const cell = event.target.closest('.cell');
            if (!cell) return;
            const col = parseInt(cell.dataset.col);
            console.log(`Player (ID: ${humanPlayerId}) clicked column: ${col}`);
             if (!currentBoard || currentBoard.length === 0) {
                 console.error("currentBoard not initialized or empty during click");
                 return;
             }
            const topRowIndex = currentBoard.length - 1;
            if (currentBoard[topRowIndex][col] !== 0) {
                updateMessage("Column is full! Try another.");
                return;
            }
            localIsPlayerTurn = false;
            setTurnIndicator();
            updateMessage("Processing your move...");
            try {
                const response = await fetch('/play', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', },
                    body: JSON.stringify({ column: col }),
                });
                 if (!response.ok) {
                    let errorMsg = `HTTP error! status: ${response.status}`;
                    try { const errorData = await response.json(); errorMsg = errorData.error || errorMsg; } catch (e) {}
                     if (response.status === 400) {
                         const data = await response.json();
                         updateMessage(data.message);
                         localIsPlayerTurn = data.is_player_turn;
                         setTurnIndicator();
                         return;
                     } else {
                        throw new Error(errorMsg);
                     }
                }
                const data = await response.json();
                console.log("Received from /play:", data);
                updateBoard(data.board);
                updateMessage(data.message);
                localGameActive = data.game_active;
                if (localGameActive && data.trigger_ai) {
                    setTimeout(requestAIMove, 50);
                } else {
                     setTurnIndicator();
                }
            } catch (error) {
                console.error('Error during play request:', error);
                updateMessage(`Error communicating with server: ${error.message}. Please refresh.`);
                localGameActive = false;
                localIsPlayerTurn = false;
                setTurnIndicator();
            }
        });
    } 

    if (newGameBtn) {
        newGameBtn.addEventListener('click', () => {
            window.location.href = '/reset';
        });
    }

    // --- Initial Setup on Load ---
    if (localGameActive && localInitialBoard) {
        console.log("Game active on load. Updating board.");
        updateBoard(localInitialBoard);
        if(gameArea) gameArea.style.display = 'flex';
        console.log("Checking if AI first move needed:", localNeedsAIFirstMove, "Game Active:", localGameActive);
        if (localNeedsAIFirstMove && localGameActive) {
            console.log("Condition TRUE. Scheduling AI first move request...");
            setTimeout(requestAIFirstMove, 100);
        } else {
            console.log("Condition FALSE. AI first move not needed or game not active on load.");
            setTurnIndicator();
        }
    } else {
         console.log("No active game found in session on load.");
         if(gameArea) gameArea.style.display = 'none';
    }

});
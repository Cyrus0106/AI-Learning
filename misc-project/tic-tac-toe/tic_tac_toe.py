# define the board as a list
board = [" " for _ in range(9)]  # 3x3 tic tac toe board represented in a 1D list


# function to print the board
def print_board(board):
    for row in [board[i * 3 : (i + 1) * 3] for i in range(3)]:
        print("|" + "|".join(row) + "|")


# function to check winner
def check_winner(board,player):
    win_conditions = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]
    for condition in win_conditions: 
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == player:
            return True
    return False

# function to check if the board is full
def is_board_full(board):
    return " " not in board

# function to evaluate the board for the minimax algorithm
def evaluate_board(board):
    if check_winner(board,"O"): # AI is 'O'
        return 1
    elif check_winner(board,"X"): # player is 'X'
        return -1
    else:
        return 0

# minimax function to calculate the best move
def minimax(board, depth, is_minimaxing):
    score = evaluate_board(board)
    if score == 1 or score == -1 or is_board_full(board):
        return score

    if is_minimaxing:  # AI's turn (maximize)
        best_score = -float("inf")
        for i in range(9):
            if board[i] == " ":
                board[i] = "O"
                best_score = max(best_score, minimax(board, depth + 1, False))
                board[i] = " "  # undo move
        return best_score
    else:  # Player's turn (minimize)
        best_score = float("inf")
        for i in range(9):
            if board[i] == " ":
                board[i] = "X"
                best_score = min(best_score, minimax(board, depth + 1, True))
                board[i] = " "  # undo move
        return best_score
# find best move for i
def find_best_move(board):
    best_value = -float("inf")
    best_move = -1

    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            move_value = minimax(board,0,False)
            board[i] = " "
            if move_value > best_value:
                best_value = move_value
                best_move = i
    return best_move

# main game loop
def play_game():
    while True:
        print_board(board)

        # player move
        player_move = int(input("Enter your move (1-9): ")) - 1
        if board[player_move] != " ":
            print("Invalid move. Try again.")
            continue
        board[player_move] = "X"

        # check if player wins
        if check_winner(board,"X"):
            print(board)
            print("You win!")
            break

        # check for a draw
        if is_board_full(board):
            print(board)
            print("Draw!")
            break

        # AI move
        ai_move = find_best_move(board)
        board[ai_move] = "O"

        # check if AI wins
        if check_winner(board,"O"):
            print(board)
            print("AI wins!")
            break

        # check for a draw
        if is_board_full(board):
            print(board)
            print("Draw!")
            break
# start the game
play_game()
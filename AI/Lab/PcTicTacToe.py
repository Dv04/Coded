# Set up the game board as a list
board = ["-", "-", "-", "-", "-", "-", "-", "-", "-"]


# Define a function to print the game board
def print_board():
    print("Welcome to Tic Tac Toe!\n")
    print("+-----------+")
    print("| " + board[0] + " | " + board[1] + " | " + board[2] + " |")
    print("| " + board[3] + " | " + board[4] + " | " + board[5] + " |")
    print("| " + board[6] + " | " + board[7] + " | " + board[8] + " |")
    print("+-----------+")



# Define a function to handle a player's turn
def take_turn(player):
    print("\n" + player + "'s turn.")

    if player == "X":
        position = input("Choose an empty position from 1-9: ")
        while position not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            position = input("Invalid input. Choose an empty position from 1-9: ")
        position = int(position) - 1
    else:
        # Computer's turn (O)
        position = get_computer_move()

    while board[position] != "-":
        if player == "X":
            position = (
                int(
                    input("Position not available. Choose an empty position from 1-9: ")
                )
                - 1
            )
        else:
            position = get_computer_move()

    board[position] = player
    print_board()


# Define a function to get the computer's move (O)
def get_computer_move():
    best_score = -float("inf")
    best_move = None

    for i in range(9):
        if board[i] == "-":
            board[i] = "O"
            score = minimax(board, 0, False)
            board[i] = "-"

            if score > best_score:
                best_score = score
                best_move = i

    return best_move


# Define the minimax algorithm
def minimax(board, depth, is_maximizing):
    scores = {"X": -1, "O": 1, "tie": 0}

    result = check_game_over(board)

    if result in scores:
        return scores[result]

    if is_maximizing:
        max_eval = -float("inf")
        for i in range(9):
            if board[i] == "-":
                board[i] = "O"
                eval = minimax(board, depth + 1, False)
                board[i] = "-"
                max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float("inf")
        for i in range(9):
            if board[i] == "-":
                board[i] = "X"
                eval = minimax(board, depth + 1, True)
                board[i] = "-"
                min_eval = min(min_eval, eval)
        return min_eval


# Define a function to check if the game is over
def check_game_over(board):
    # Check for a win
    for combo in [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != "-":
            return board[combo[0]]  # Return the winner

    # Check for a tie
    if "-" not in board:
        return "tie"

    # Game is not over
    return "play"


# Define the main game loop
def play_game():
    print(
        "The board is numbered 1-9, starting from top left and going left to right, top to bottom.\n"
    )
    print_board()
    current_player = "X"
    game_over = False
    while not game_over:
        take_turn(current_player)
        game_result = check_game_over(board)
        if game_result == "X" or game_result == "O":
            print(game_result + " wins!")
            game_over = True
        elif game_result == "tie":
            print("It's a tie!")
            game_over = True
        else:
            # Switch to the other player
            current_player = "O" if current_player == "X" else "X"


if __name__ == "__main__":
    # Start the game
    play_game()

# Set up the game board as a list
board = ["-", "-", "-",
		"-", "-", "-",
		"-", "-", "-"]

# Define a function to print the game board
def print_board():
  print("Welcome to Tic Tac Toe!\n")
  print(board[0] + " | " + board[1] + " | " + board[2])
  print(board[3] + " | " + board[4] + " | " + board[5])
  print(board[6] + " | " + board[7] + " | " + board[8])

# Define a function to handle a player's turn
def take_turn(player):
	print("\n" + player + "'s turn.")
	position = input("Choose an empty position from 1-9: ")
	while position not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
		position = input("Invalid input. Choose an empty position from 1-9: ")
	position = int(position) - 1
	while board[position] != "-":
		position = int(input("Position not available. Choose an empty position from 1-9: ")) - 1
	board[position] = player
	print_board()

# Define a function to check if the game is over
def check_game_over():
	# Check for a win
	if (board[0] == board[1] == board[2] != "-") or \
	(board[3] == board[4] == board[5] != "-") or \
	(board[6] == board[7] == board[8] != "-") or \
	(board[0] == board[3] == board[6] != "-") or \
	(board[1] == board[4] == board[7] != "-") or \
	(board[2] == board[5] == board[8] != "-") or \
	(board[0] == board[4] == board[8] != "-") or \
	(board[2] == board[4] == board[6] != "-"):
		return "win"
	# Check for a tie
	elif "-" not in board:
		return "tie"
	# Game is not over
	else:
		return "play"

# Define the main game loop
def play_game():
    print("The board is numbered 1-9, starting from top left and going left to right, top to bottom.\n")
    print_board()
    current_player = "X"
    game_over = False
    while not game_over:
        take_turn(current_player)
        game_result = check_game_over()
        if game_result == "win":
            print(current_player + " wins!")
            game_over = True
        elif game_result == "tie":
            print("It's a tie!")
            game_over = True
        else:
            # Switch to the other player
            current_player = "O" if current_player == "X" else "X"
   
if __name__ == "__main__":
    play_game()

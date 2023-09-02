from TicTacToe import play_game as pg
from PcTicTacToe import play_game as pc

print("Hello Player\nDo you want to play Tic Tac Toe? (Y/N)")
ans = input()
if ans == "Y" or ans == "y":
    print("Would you like to play against a computer or another player? (C/P)")
    ans = input()
    if ans == "C" or ans == "c":
        pc()
    elif ans == "P" or ans == "p":
        pg()
    else:
        print("Invalid input")
elif ans == "N" or ans == "n":
    print("Goodbye")
else:
    print("Invalid input")

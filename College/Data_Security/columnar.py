import numpy as np
import math


class Color:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


MAT_SIZE = 5
inp = input(Color.YELLOW + "\n\nEnter the plain Text: " + Color.RESET).lower()
inp = list(inp)
print(Color.GREEN + str(inp) + Color.RESET)
while len(inp) > MAT_SIZE * MAT_SIZE:
    MAT_SIZE += 1

key = input(Color.YELLOW + "\n\nEnter the key: " + Color.RESET)
key = list(map(int, key))
key = list(dict.fromkeys(key))
print(Color.CYAN + "\nKey: " + str(key) + Color.RESET)
if len(key) != MAT_SIZE:
    print(Color.RED + "\n\nInvalid key size. Key size must be 5." + Color.RESET)
    exit(0)

key_inv = []
for i in range(len(key)):
    key_inv.append(key.index(i))

print(Color.YELLOW + "Key inverse: " + str(key_inv) + Color.RESET)

mat = np.zeros((MAT_SIZE, MAT_SIZE), dtype=str)
for i in range(MAT_SIZE):
    for j in range(MAT_SIZE):
        if inp != None and len(inp) > 0:
            mat[i][j] = inp.pop(0)
        else:
            mat[i][j] = " "
print(
    Color.MAGENTA
    + "\n\nThe matrix created from this plain text will be:\n"
    + str(mat)
    + Color.RESET
)

cypher = ""
for i in range(MAT_SIZE):
    for j in range(MAT_SIZE):
        cypher += mat[i][key[j]]
print(Color.BLUE + "\nCypher: " + cypher + Color.RESET)
cypher = list(cypher)

for i in range(MAT_SIZE):
    for j in range(MAT_SIZE):
        if cypher != None and len(cypher) > 0:
            mat[i][j] = cypher.pop(0)
        else:
            mat[i][j] = " "
print(
    Color.MAGENTA
    + "\n\nThe matrix created from this cypher will be:\n"
    + str(mat)
    + Color.RESET
)

decrypted = ""
for i in range(MAT_SIZE):
    for j in range(MAT_SIZE):
        decrypted += mat[i][key_inv[j]]
print(Color.GREEN + "\nDecrepyted plain text is: " + decrypted + Color.RESET + "\n\n")

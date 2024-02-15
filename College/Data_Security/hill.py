import numpy as np
from sympy import Matrix


def encode(message, K):
    message = [ord(i.lower()) - 97 for i in message]
    while len(message) % 2 != 0:
        message.append(0)
    message = np.array(message).reshape(-1, 2)
    print("\nEncoded message: \n", message, "\n")
    cipher = ""
    for i in message:
        print("Printing: ", (np.dot(K, i)).tolist())
        cipher += "".join([chr(int(x) % 26 + 97) for x in np.dot(K, i).tolist()])
    return cipher


def decode(cipher, K_inv):
    cipher = [ord(i.lower()) - 97 for i in cipher]
    cipher = np.array(cipher).reshape(-1, 2)
    print("\nCipher : \n", cipher, "\n")
    message = ""
    for i in cipher:
        message += "".join([chr(int(x) % 26 + 97) for x in np.dot(K_inv, i).tolist()])
    return message


def generate_key_matrix(keyword):
    keyword = keyword.lower()
    keyword = keyword.replace(" ", "")
    keyword = list(keyword)
    keyword_length = len(keyword)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    alphabet = list(alphabet)
    for char in keyword:
        if char in alphabet:
            alphabet.remove(char)
    keyword.extend(alphabet)
    keyword_matrix = Matrix(
        [
            [ord(keyword[0]) - 97, ord(keyword[1]) - 97],
            [ord(keyword[2]) - 97, ord(keyword[3]) - 97],
        ]
    )
    return keyword_matrix


# User input for keyword
keyword = input("\nEnter the keyword: ")
K = generate_key_matrix(keyword)
print("\nKey Matrix: ", K)

message = input("\nEnter the message: ")
cipher = encode(message, K)
print("\nCipher: " + cipher)

K_inv = K.inv_mod(26)
print("\nInverse Matrix: ", K_inv)
message_decoded = decode(cipher, K_inv)
print("\nDecoded message: " + message_decoded)

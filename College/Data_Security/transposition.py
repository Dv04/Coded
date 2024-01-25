# Implement the Transposition encyption and decyption methods for plain text to ciphers. Following 2 methods are to be used:
# Swapping the characters in plain text
# Trailing

import math


def swap_chars(text, i, j):
    text_list = list(text)
    text_list[i], text_list[j] = text_list[j], text_list[i]
    return "".join(text_list)


def swap_encrypt(text):
    for i in range(0, len(text) - 1, 2):
        text = swap_chars(text, i, i + 1)
    return text


def swap_decrypt(text):
    return swap_encrypt(text)  # The operation is symmetric


def rail_fence_encrypt(text, key):
    cipher = ""
    length = len(text)
    for i in range(key):
        step = key * 2 - 2
        for j in range(i, length, step):
            cipher += text[j]
            if i != 0 and i != key - 1 and j + step - 2 * i < length:
                cipher += text[j + step - 2 * i]
    return cipher


def rail_fence_decrypt(cipher, key):
    length = len(cipher)
    plain = [""] * length
    count = 0
    for i in range(key):
        step = key * 2 - 2
        for j in range(i, length, step):
            plain[j] = cipher[count]
            count += 1
            if i != 0 and i != key - 1 and j + step - 2 * i < length:
                plain[j + step - 2 * i] = cipher[count]
                count += 1
    return "".join(plain)


text = "Hello everyone this is Dev"
key = int(input("Enter the key: "))

swap_encrypted = swap_encrypt(text)
print("Swap Encrypted: ", swap_encrypted, "\n")

swap_decrypted = swap_decrypt(swap_encrypted)
print("Swap Decrypted: ", swap_decrypted, "\n")

encrypted = rail_fence_encrypt(text, key)
print("Encrypted: ", encrypted, "\n")

decrypted = rail_fence_decrypt(encrypted, key)
print("Decrypted: ", decrypted, "\n")


def reverse_swap_encrypt(text):
    text = text[::-1]  # Reverse the text
    return swap_encrypt(text)


def reverse_swap_decrypt(text):
    text = swap_decrypt(text)
    return text[::-1]  # Reverse the text back


def reverse_rail_fence_encrypt(text, key):
    text = text[::-1]  # Reverse the text
    return rail_fence_encrypt(text, key)


def reverse_rail_fence_decrypt(cipher, key):
    text = rail_fence_decrypt(cipher, key)
    return text[::-1]  # Reverse the text back


reverse_swap_encrypted = reverse_swap_encrypt(text)
print("Reverse Swap Encrypted: ", reverse_swap_encrypted, "\n")

reverse_swap_decrypted = reverse_swap_decrypt(reverse_swap_encrypted)
print("Reverse Swap Decrypted: ", reverse_swap_decrypted, "\n")

reverse_encrypted = reverse_rail_fence_encrypt(text, key)
print("Reverse Encrypted: ", reverse_encrypted, "\n")

reverse_decrypted = reverse_rail_fence_decrypt(reverse_encrypted, key)
print("Reverse Decrypted: ", reverse_decrypted, "\n")

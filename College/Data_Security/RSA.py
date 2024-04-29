# ANSI escape codes for some colors
RED = "\033[1;31m"
GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
RESET = "\033[0m"


def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


def is_prime(n):
    if n == 2:
        return True
    if n < 2 or n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def generate_keypair(p, q):
    if not (is_prime(p) and is_prime(q)):
        raise ValueError(RED + "Both numbers must be prime." + RESET)
    elif p == q:
        raise ValueError(RED + "p and q cannot be equal." + RESET)
    n = p * q
    phi = (p - 1) * (q - 1)

    e = 2
    while gcd(e, phi) != 1:
        e += 1

    d = 0
    while (d * e) % phi != 1:
        d += 1

    return ((e, n), (d, n))


def encrypt(public_key, message):
    e, n = public_key
    cipher = [pow(ord(char), e, n) for char in message]
    return cipher


def decrypt(private_key, cipher):
    d, n = private_key
    message = [chr(pow(char, d, n)) for char in cipher]
    return "".join(message)


def main():
    p = 17
    q = 19
    public_key, private_key = generate_keypair(p, q)
    print(GREEN + "Public key:" + RESET, public_key)
    print(BLUE + "Private key:" + RESET, private_key)
    message = "Dev Sanghvi"
    cipher = encrypt(public_key, message)
    decrypted = decrypt(private_key, cipher)
    print(CYAN + "Original message:" + RESET, message)
    print(RED + "Encrypted message:" + RESET, "".join(map(str, cipher)))
    print(GREEN + "Decrypted message:" + RESET, decrypted)


if __name__ == "__main__":
    main()

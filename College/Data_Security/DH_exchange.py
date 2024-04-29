import random
import math

# ANSI escape codes for some colors
RED = "\033[1;31m"
GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
RESET = "\033[0m"


def is_prime(n):
    if n == 2:
        return True
    if n % 2 == 0 or n == 1:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def get_prime():
    while True:
        n = random.randint(100, 1000)
        if is_prime(n):
            return n


def get_base(prime):
    while True:
        base = random.randint(2, prime - 1)
        if math.gcd(base, prime) == 1:
            return base


def get_secret_key(prime):
    return random.randint(2, prime - 1)


def get_public_key(base, secret_key, prime):
    return (base**secret_key) % prime


def get_shared_secret_key(public_key, secret_key, prime):
    return (public_key**secret_key) % prime


def main():
    prime = get_prime()
    base = get_base(prime)
    print(GREEN + "Prime number:" + RESET, prime)
    print(GREEN + "Base:" + RESET, base)
    secret_key_alice = get_secret_key(prime)
    secret_key_bob = get_secret_key(prime)
    print(BLUE + "\nSecret key of Alice:" + RESET, secret_key_alice)
    print(BLUE + "Secret key of Bob:" + RESET, secret_key_bob)
    public_key_alice = get_public_key(base, secret_key_alice, prime)
    public_key_bob = get_public_key(base, secret_key_bob, prime)
    print(CYAN + "\nPublic key of Alice:" + RESET, public_key_alice)
    print(CYAN + "Public key of Bob:" + RESET, public_key_bob)
    shared_secret_key_alice = get_shared_secret_key(
        public_key_bob, secret_key_alice, prime
    )
    shared_secret_key_bob = get_shared_secret_key(
        public_key_alice, secret_key_bob, prime
    )
    print(RED + "\nShared secret key of Alice:" + RESET, shared_secret_key_alice)
    print(RED + "Shared secret key of Bob:" + RESET, shared_secret_key_bob)


if __name__ == "__main__":
    main()

def prime_list(limit):
    primes = [True] * (limit + 1)
    primes[0] = primes[1] = False

    for i in range(2, int(limit**0.5) + 1):
        if primes[i]:
            for j in range(i * i, limit + 1, i):
                primes[j] = False

    return [num for num, is_prime in enumerate(primes) if is_prime]


def is_prime(num):
    if num < 2:
        return False
    if num == 2 or num == 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False

    i = 5
    w = 2
    itr = 0

    while i * i <= num:
        if num % i == 0:
            print(itr)
            return False
        i += w
        w = 6 - w
        itr += 1

    print(itr)
    return True


# print(prime_list(10000000))
print(is_prime(9973))

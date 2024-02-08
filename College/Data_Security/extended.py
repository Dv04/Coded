# Make the code of extended GCD which also calculates the multiplicative inverse of a number.
# Algorithm:
# 1. Take two numbers as input. the numbers are m and b, where m is the modulo and b is the number.
# 2. define a function extended_gcd(m, b) which takes two numbers as input.
# 3. define A1, A2, A3, B1, B2, B3, T1, T2, T3, Q as variables.
# 4. A1 = 1, A2 = 0, A3 = m, B1 = 0, B2 = 1, B3 = b
# 5. while B3 != 0 or B3 != 1:
# 6. Q = A3 // B3
# 7. T1 = A1 - (Q * B1), T2 = A2 - (Q * B2), T3 = A3 - (Q * B3)
# 8. A1 = B1, A2 = B2, A3 = B3, B1 = T1, B2 = T2, B3 = T3
# 9. if B3 == 0:
# 9-1. return A3, None
# 10. if B3 == 1:
# 10-1. return B3, B2


def extended_gcd(m, b):
    A1, A2, A3, B1, B2, B3, T1, T2, T3, Q = 1, 0, m, 0, 1, b, 0, 0, 0, 0
    iteration = 0
    print()
    while B3 != 0 and B3 != 1:
        print(
            "\033[31mQ:\033[0m {:<8} \033[33mA1:\033[0m {:<8} \033[32mA2:\033[0m {:<8} \033[36mA3:\033[0m {:<8} \033[34mB1:\033[0m {:<8} \033[35mB2:\033[0m {:<8} \033[37mB3:\033[0m {:<8} \033[35mIteration:\033[0m {:<8}".format(
                Q, A1, A2, A3, B1, B2, B3, iteration
            )
        )
        Q = A3 // B3
        T1 = A1 - (Q * B1)
        T2 = A2 - (Q * B2)
        T3 = A3 - (Q * B3)
        A1, A2, A3, B1, B2, B3 = B1, B2, B3, T1, T2, T3
        iteration += 1
    print(
        "\033[31mQ:\033[0m {:<8} \033[33mA1:\033[0m {:<8} \033[32mA2:\033[0m {:<8} \033[36mA3:\033[0m {:<8} \033[34mB1:\033[0m {:<8} \033[35mB2:\033[0m {:<8} \033[37mB3:\033[0m {:<8} \033[35mIteration:\033[0m {:<8}".format(
            Q, A1, A2, A3, B1, B2, B3, iteration
        )
    )
    if B3 == 0:

        return A3, None, None
    if B3 == 1:
        while B2 < 0:
            B2 += m
        while B1 < 0:
            B1 += b
        return B3, B2, B1


a, b = input("\n\033[94mEnter two numbers: \033[0m").split()
a, b = int(a), int(b)
d, x, y = extended_gcd(a, b)


if d != 1:
    print(
        "\n\033[91mThe numbers are not co-prime. The multiplicative inverse does not exist.\033[0m\n"
    )
else:
    print("\n\033[92mThe numbers are co-prime\033[0m")

    while x < 0:
        x += b
    while y < 0:
        y += a

    print("\n\033[93mGCD of {} and {} is {}\033[0m".format(a, b, d))
    print(
        "\n\033[93mThe multiplicative inverse of {} mod {} is {}\033[0m".format(a, b, x)
    )
    print(
        "\033[93mThe multiplicative inverse of {} mod {} is {}\033[0m\n".format(b, a, y)
    )

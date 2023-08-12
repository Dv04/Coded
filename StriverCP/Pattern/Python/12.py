a = int(input())

for i in range(a):
    for j in range(i):
        print(j + 1, end="")
    for _ in range(2 * (a - i - 1)):
        print(" ", end="")
    for j in range(i, 0, -1):
        print(j, end="")
    print()

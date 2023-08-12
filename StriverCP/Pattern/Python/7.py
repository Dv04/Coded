a = int(input())

for i in range(a):
    for _ in range(a - i):
        print(" ", end="")
    for _ in range(i + 1):
        print("*", end="")
    for _ in range(i):
        print("*", end="")

    print()

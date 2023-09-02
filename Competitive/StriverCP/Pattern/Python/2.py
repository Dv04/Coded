a = int(input())

for i in range(a):
    for _ in range(i + 1):
        print("*", end="")
    print()

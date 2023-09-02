a = int(input())

for i in range(a):
    for _ in range(i ):
        print("*", end="")
    print()

for i in range(a ):
    for _ in range(a - i):
        print("*", end="")
    print()

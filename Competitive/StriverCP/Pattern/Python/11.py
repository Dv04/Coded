a = int(input())

for i in range(a):
    for j in range(i + 1):
        print("1" if (i + j) % 2 == 0 else "0", end="")
    print()

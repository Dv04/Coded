a = int(input())
n = 1
for i in range(a):
    for j in range(i + 1):
        print(n, " ", end="")
        n += 1
    print()

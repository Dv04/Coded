a = int(input())
n = 'A'
for i in range(a):
    for j in range(i + 1):
        print(n, " ", end="")
        n = chr(ord(n) + 1)
    print()

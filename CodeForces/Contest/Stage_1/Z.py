a, b, c, d = map(int, input().split())

def ln(x):
    n = 1000.0
    print(n * ((x ** (1/n)) - 1))
    return n * ((x ** (1/n)) - 1)

if b*ln(a)>d*ln(a):
    print("YES")
else:
    print("NO")
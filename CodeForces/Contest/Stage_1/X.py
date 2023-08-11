a, b, c, d = map(int, input().split())
if c<=b and a<=d:
    print(max(c,a), min(b,d))
else:
    print("-1")
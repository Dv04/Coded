t = int(input())
for i in range(t):
    a,b,c = map(int, input().split())
    print("YES" if max(a,b,c) == a+b+c-max(a,b,c) else "NO")
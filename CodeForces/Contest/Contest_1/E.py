a, b = map(int, input().split())
if a-b <=1 and a-b >=-1 and (a != 0 or b != 0):
    print("YES")
else:
    print("NO")
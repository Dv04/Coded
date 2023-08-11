a = input().split()
a[0], a[2], a[4] = int(a[0]), int(a[2]), int(a[4])
if a[1] == '+':
    print("Yes" if (a[0] + a[2] == a[4]) else (a[0] + a[2]))
elif a[1] == '-':
    print("Yes" if (a[0] - a[2] == a[4]) else (a[0] - a[2]))
elif a[1] == '*':
    print("Yes" if (a[0] * a[2] == a[4]) else (a[0] * a[2]))
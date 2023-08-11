a = input().split()
a[0], a[2] = int(a[0]), int(a[2])
if a[1] == '>':
    print("Right" if a[0] > a[2] else "Wrong")
elif a[1] == '<':
    print("Right" if a[0] < a[2] else "Wrong")
elif a[1] == '=':
    print("Right" if a[0] == a[2] else "Wrong")
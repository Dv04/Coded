t = int(input())
for i in range(t):
    a = int(input())
    if a < 1400:
        print("Division 4")
    elif a < 1600:
        print("Division 3")
    elif a < 1900:
        print("Division 2")
    else:
        print("Division 1")
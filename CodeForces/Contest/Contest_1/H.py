a, b, c = map(int, input().split())
if (a * b) % c == 0:
    if ((a / c) * b) <= 2147483647 and ((a / c) * b) >= -2147483648:
        print("int")
    else:
        print("long long")
else:
    print("double")

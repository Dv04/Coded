a = input()
if a==a[::-1]:
    print(f"{a}\nYES")
else:
    print(f"{int(a[::-1])}\nNO")
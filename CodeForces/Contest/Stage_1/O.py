a = input()
first = 0
for i in a:
    if i != "+" and i != "-" and i != "*" and i != "/":
        first *= 10
        first += int(i)
    else:
        if i == "+":
            res = first + int(a[a.index(i)+1:])
        elif i == "-":
            res = first - int(a[a.index(i)+1:])
        elif i == "*":
            res = first * int(a[a.index(i)+1:])
        elif i == "/":
            res = first / int(a[a.index(i)+1:])
    
print(int(res))
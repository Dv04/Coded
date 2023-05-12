n = int(input())
b = 0
for i in range(n):
    a = list(input())
    c = 0
    for j in a:
        if j == 0:
            c += 1
    if c < 2:
        b+=1

print(b)
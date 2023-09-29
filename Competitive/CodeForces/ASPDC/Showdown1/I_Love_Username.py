n = int(input())
a = [int(x) for x in input().split()]
min = a[0]
max = a[0]
count = 0
for i in range(1,n):
    if a[i] > max:
        max = a[i]
        count += 1
    elif a[i] < min:
        min = a[i]
        count += 1
print(count)
a = int(input())
for i in range (2,a):
    print(f"{i} " if all(i % j != 0 for j in range(2, int(i**0.5)+1)) else '',end='')
print()
a = input()
b = int(input())
c = [i for i in map(int, input().split())]
for i in c:
    print(a*i)

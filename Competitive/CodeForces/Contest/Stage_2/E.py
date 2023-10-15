t = int(input())
x=0
for i in range(t):
    a=input()
    if a[1]=="+":
        x+=1
    else:
        x-=1
print(x)
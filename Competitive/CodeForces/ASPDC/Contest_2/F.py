a,b = map(int,input().split())
for i in range(b):
    if a%10==0:
        a/=10
    else:
        a-=1
print(int(a))
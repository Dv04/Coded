a = int(input())
b = [a for a in map(int,input().split())]
o,e,p,n = 0,0,0,0
for i in b:
    if i <0:
        n+=1
    elif i>0:
        p+=1
    if i%2==0:
        e+=1
    else:
        o+=1

print("Even:",e)
print("Odd:",o)
print("Positive:",p)
print("Negative:",n)
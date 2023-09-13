# cook your dish here
for i in range(int(input())):
    a,b,c,d = map(int,input().split())
    print("YES" if (a+c)-(b+d)<0 else "NO")

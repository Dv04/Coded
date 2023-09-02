for _ in range(int(input())):
    p=int(input())
    q=list(map(int,input().split()))
    c=0
    for i in q:
        c+=i
    if p%2==1:
        if c%2==1:
            print("CHEF")
        else:
            print("CHEFINA")
    else:
        x=min(q)
        if x==1:
            print("CHEF")
            continue
        if c%2==0 and x%2==0:
            print("CHEFINA")
        else:
            print("CHEF")
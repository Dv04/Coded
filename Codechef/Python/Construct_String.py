# cook your dish here
t = int(input())



for i in range(t):
    count = 1
    leng = int(input())
    ss = str(input())
    ans = ""
    for i in range(len(ss)-1):
        if ss[i] != ss[i+1]:
            if count%2 != 0:
                ans += ss[i]
            else:
                ans += 2*ss[i]
            count = 1
        else:
            count += 1
    if count%2 != 0:
        ans += ss[i]
    else:
        ans += 2*ss[i]
    print(ans)
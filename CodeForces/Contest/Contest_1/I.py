a = input()
if a[0] == '0' or a[1] == '0':
    print("YES")
elif int(a[0])%int(a[1]) == 0 or int(a[1])%int(a[0]) == 0:
    print("YES")
else:
    print("NO")
t = int(input())
for i in range(t):
    a = input()
    print("YES" if int(a[0])+int(a[1])+int(a[2]) == int(a[3])+int(a[4])+int(a[5]) else "NO")
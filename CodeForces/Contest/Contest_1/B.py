a, b, c = map(int, input().split())
if a%c==0 or b%c==0:
    if b%c!=0:
        print("Memo")
    elif a%c!=0:
        print("Momo")
    else:
        print("Both")
else:
    print("No One")
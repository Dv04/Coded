a = input().split(".")
if int(a[1])==0:
    print("int",int(a[0]))
else:
    print("float {} 0.{}".format(int(a[0]),int(a[1])))
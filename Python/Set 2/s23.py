inp = int(input("Enter a Number: "))
stlen = str(inp)

revnum = 0
for i in range(len(stlen)):
    revnum += inp%10
    inp /= 10
    inp = int(inp)
    # print(revnum)
    revnum *= 10

revnum/=10
print(int(revnum))
    
# cook your dish here
for i in range(int(input())):
    a = int(input())
    b = input()
    for i in range(len(b)-2):
        if i=='1':
            b[i:i+3]='100'
    print(b)
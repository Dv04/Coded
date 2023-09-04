a, b = map(int,input("Please enter two numbers: ").split())
while b:
    a, b = b, a%b
print(a)

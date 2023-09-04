n, name = map(str,input("Enter a number and a name: ").split())
n = int(n)

def printer(n):
    if n == 0:
        return
    printer(n-1)
    print(name, sep = ' ')
    
printer(n)
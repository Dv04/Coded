# Print 1 to N using recursion.

def printer(n):
    if n == 0:
        return
    printer(n-1)
    print(n, sep = ' ')
    
printer(int(input("Enter a number: ")))

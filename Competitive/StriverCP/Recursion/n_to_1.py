def printer(n):
    if n == 0:
        return
    print(n, sep = ' ')
    printer(n-1)

printer(int(input("Enter a number: ")))
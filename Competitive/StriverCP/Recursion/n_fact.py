# Print the factorial of first n numbers using recursion

def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n-1)

print(factorial(int(input("Enter a number: "))))
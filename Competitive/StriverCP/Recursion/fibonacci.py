# Print Fibonacci of n numbers using recursion.

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(int(input("Enter a number: "))))
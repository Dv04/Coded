# Print the summation of first n numbers using recursion

def summation(n):
    if n == 0:
        return 0
    return n + summation(n-1)

print(summation(int(input("Enter a number: "))))
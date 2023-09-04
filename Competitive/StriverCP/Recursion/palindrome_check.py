# check palindrome using recursion.

def palindrome(string):
    if len(string) <= 1:
        return True
    if string[0] != string[-1]:
        return False
    return palindrome(string[1:-1])

print(f"{palindrome(input('Enter a string: '))}")
# An integer n is strictly palindromic if, for every base b between 2 and n - 2 (inclusive), the string representation of the integer n in base b is palindromic.

# Given an integer n, return true if n is strictly palindromic and false otherwise.

# A string is palindromic if it reads the same forward and backward.

class Solution:
    def isStrictlyPalindromic(self, n: int):

        number_list = list(str(n))
        number_list.reverse()
        print("Reversed Number is: " + ''.join(number_list))


print(Solution().isStrictlyPalindromic(32))

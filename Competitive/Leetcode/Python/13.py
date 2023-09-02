# Given a roman numeral, convert it to an integer.

class Solution:
    def romanToInt(self, s: str) -> int:
        roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        for i in range(len(s)):
            if i > 0 and roman[s[i]] > roman[s[i - 1]]:
                total += roman[s[i]] - 2 * roman[s[i - 1]] 
            else:
                total += roman[s[i]]
        return total
    
trial = Solution()
print(trial.romanToInt('MCMXCIV'))


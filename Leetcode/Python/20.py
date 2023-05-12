from collections import deque

class Solution:
    def isValid(self, s: str) -> bool:
        
        stack = deque()
        
        for i in s:
            if (i =='(' or i == '[' or i == '{'):
                stack.append(i)
            else:
                if i == stack[-1]:
                    stack.pop()
                    
                else:
                    return False
        if len(stack) == 0:
            return True
        else:
            return False
    
print(Solution().isValid("()[]{}"))
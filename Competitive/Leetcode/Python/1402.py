'''
#class Solution:
#     def maxSatisfaction(self, satisfaction: list[int]) -> int:
            
#         satisfaction.sort(reverse=True)
#         s = dishSum = 0

#         for dish in satisfaction:
#             dishSum += dish
#             if dishSum <= 0:
#                 break
#             s += dishSum
        
#         return s
'''
import itertools
class Solution:
    def maxSatisfaction(self, satisfaction: list[int]) -> int:
        return max([0]+list(itertools.accumulate(itertools.accumulate(sorted(satisfaction)[::-1]))))    
    
print(Solution().maxSatisfaction([-1,-8,0,5,-9]))
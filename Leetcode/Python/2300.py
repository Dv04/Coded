# You are given two positive integer arrays spells and potions, of length n and m respectively, where spells[i] represents the strength of the ith spell and potions[j] represents the strength of the jth potion.

# You are also given an integer success. A spell and potion pair is considered successful if the product of their strengths is at least success.

# Return an integer array pairs of length n where pairs[i] is the number of potions that will form a successful pair with the ith spell.

import numpy as np
import math


class Solution:

    def successfulPairs(self, spells: list[int], potions: list[int], success: int) -> list[int]:
        potions.sort()
        ans = []

        def bi_search(nums: list[int], success: int) -> int:
            target = (success + spell - 1) // spell
            lo, hi = 0, len(potions)
            while lo < hi:
                mid = lo + (hi - lo) // 2
                if potions[mid] >= target:
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        
        for spell in spells:
            ans.append(len(potions)-bi_search(potions, success))
        return ans


print(Solution().successfulPairs(
    [15, 8, 19], [38, 36, 23], 328))

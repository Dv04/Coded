# You are given a 0-indexed array nums consisting of positive integers.
# There are two types of operations that you can apply on the array any number of times:
# Choose two elements with equal values and delete them from the array.
# Choose three elements with equal values and delete them from the array.
# Return the minimum number of operations required to make the array empty, or -1 if it is not possible.

from math import ceil
from typing import List
import collections

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        freq = collections.Counter(nums)
        print(freq)
        if 1 in freq.values():
            return -1
        count = 0
        for i in freq.values():
            count+=ceil(i/3)
        return count

s = Solution()
print(s.minOperations(nums = [2,1,2,2,3,3]))
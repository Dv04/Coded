# You are given an array nums of positive integers and an integer k.
# In one operation, you can remove the last element of the array and add it to your collection.
# Return the minimum number of operations needed to collect elements 1, 2, ..., k.


class Solution:
    from typing import List

    def minOperations(self, nums: List[int], k: int) -> int:
        dicti = {}
        for i in range(1, k + 1):
            dicti[i] = 1
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] in dicti.keys():
                dicti[nums[i]] = 0
            if 1 not in dicti.values():
                return len(nums) - i


s = Solution()
print(s.minOperations(nums=[3, 1, 5, 4, 2], k=5))  # Output: 2

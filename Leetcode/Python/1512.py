class Solution:
    def numIdenticalPairs(self, nums: list[int]) -> int:
        return sum((1 if nums[x] == nums[y] else 0 )for x in range(len(nums)) for y in range(x,len(nums)))
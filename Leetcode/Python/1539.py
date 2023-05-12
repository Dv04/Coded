class Solution:
    def findKthPositive(self, arr: list[int], k: int) -> int:
        change = 1
        while True:
            if (change not in arr):
                k-=1
            if (k == 0):
                return change
            change += 1
print(Solution().findKthPositive([2,3,4,7,11], 5))
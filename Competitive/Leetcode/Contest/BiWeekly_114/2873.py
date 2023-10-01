from typing import List


class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        if len(nums) == 3:
            return (
                ((nums[0] - nums[1]) * nums[2])
                if ((nums[0] - nums[1]) * nums[2]) > 0
                else 0
            )
        m = max(nums[0 : len(nums) - 2])
        s = min(nums[(nums.index(m) + 1) : len(nums) - 1])
        t = max(nums[(nums.index(s) + 1) : len(nums)])
        t1 = max(nums[2 : len(nums)])
        s1 = min(nums[1 : (nums.index(t1))])

        m1 = max(nums[0 : nums.index(s1, 1)])
        if ((m1 - s1) * t1) < ((m - s) * t):
            if m < s:
                return 0
            print(m, s, t)
            return (m - s) * t
        else:
            if m1 < s1:
                return 0
            print(m1, s1, t1, "1")
            return (m1 - s1) * t1


s = Solution()
print(s.maximumTripletValue([1, 19, 1, 3, 18, 10, 16, 9, 3, 17, 8, 9]))

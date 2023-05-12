# Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.

class Solution:
    def countBits(self, n: int) -> list[int]:
        ans = [0]
        for i in range(1, n+1):
            ans.append(ans[i//2] + i % 2)
        return ans


print(Solution.countBits(Solution, 10000))

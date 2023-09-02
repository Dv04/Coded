# Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have gone and will come back in h hours.

# Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, she eats all of them instead and will not eat any more bananas during this hour.

# Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

# Return the minimum integer k such that she can eat all the bananas within h hours.

import math as M


class Solution:
    def minEatingSpeed(self, piles: list[int], h: int) -> int:

        lo, hi = 1, max(piles)

        def f(x):
            return sum(M.ceil(t/x) for t in piles) <= h

        while lo < hi:
            mid = (lo + hi) // 2
            if not f(mid):
                lo = mid + 1
            else:
                hi = mid
        return lo


print(Solution().minEatingSpeed([30, 11, 23, 4, 20], 9))

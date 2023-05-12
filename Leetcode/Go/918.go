// Given a circular integer array nums of length n, return the maximum possible sum of a non-empty subarray of nums.

// A circular array means the end of the array connects to the beginning of the array. Formally, the next element of nums[i] is nums[(i + 1) % n] and the previous element of nums[i] is nums[(i - 1 + n) % n].

// A subarray may only include each element of the fixed buffer nums at most once. Formally, for a subarray nums[i], nums[i + 1], ..., nums[j], there does not exist i <= k1, k2 <= j with k1 % n == k2 % n.

package main

import "fmt"

func maxSubarraySumCircular(nums []int) int {
	total = 0
	maxSum = -inf
	curMax = 0
	minSum = inf
	curMin = 0
	for num in nums:
		curMax = max(curMax + num, num)
		maxSum = max(maxSum, curMax)
		curMin = min(curMin + num, num)
		minSum = min(minSum, curMin)
		total += num
	return max(maxSum, total - minSum) if maxSum > 0 else maxSum
}

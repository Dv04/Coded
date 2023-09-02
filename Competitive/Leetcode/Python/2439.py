# You are given a 0-indexed array nums comprising of n non-negative integers.

# In one operation, you must:

# Choose an integer i such that 1 <= i < n and nums[i] > 0.
# Decrease nums[i] by 1.
# Increase nums[i - 1] by 1.
# Return the minimum possible value of the maximum integer of nums after performing any number of operations.


def minimizeArrayValue(nums: list[int]) -> int:
        
    diff = [(nums[i]-nums[i-1]) for i in range(1,len(nums))]
    for i in range(20):
        print(diff)
        print(nums)
        ind = diff.index(max(diff))
        nums[ind] += 1
        diff[ind] -= 2
        if ind == 0:
            nums[ind + 1] -= 1
            diff[ind + 1] += 1
        elif ind == len(diff)-2:
            diff[ind - 1] += 1
        else:
            nums[ind + 1] -= 1
            diff[ind + 1] += 1
            diff[ind - 1] += 1
        # try:
        #     nums[ind+1] -= 1
        # except IndexError:
        #     pass
        # try:
        #     diff[ind+1] += 1
        # except IndexError:
        #     pass
        # try:
        #     diff[ind-1] += 1
        # except IndexError:
        #     pass
        
        
        
minimizeArrayValue([3,7,13,18])
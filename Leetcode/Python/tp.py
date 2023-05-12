class Solution:
    def search(self, nums: List[int], target: int) -> int:
        def binary(fir: int, las: int):
            if fir > las:
                return -1
            mid = int((fir+las)/2)
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                return binary(mid+1, las)
            else:
                return binary(fir, mid-1)

        index = binary(0, len(nums)-1)
        if index != -1:
            return index
        else:
            return -1
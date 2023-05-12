// Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

// You must write an algorithm with O(log n) runtime complexity.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int search(int *nums, int numsSize, int target)
{
    int left = 0;
    int right = numsSize - 1;
    int mid = 0;
    while (left <= right)
    {
        mid = (left + right) / 2;
        if (nums[mid] == target)
        {
            return mid;
        }
        else if (nums[mid] < target)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }
    return -1;
}

int main(int argc, char *argv[])
{
    int nums[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int numsSize = sizeof(nums) / sizeof(nums[0]);
    int target = 5;
    int result = search(nums, numsSize, target);
    printf("result = %d\n", result);
    return 0;
}
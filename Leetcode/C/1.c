/**
 * @file 1.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief 1st problem
 * @version 1.0
 * @date 2022-11-28
 *
 * @copyright Copyright (c) 2022
 *
 */

/*
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.
*/

#include <stdio.h>
#include <stdlib.h>

int *twoSum(int *nums, int numsSize, int target, int *returnSize)
{
    int i, j;
    int *result = (int *)malloc(2 * sizeof(int));
    for (i = 0; i < numsSize; i++)
    {
        for (j = i + 1; j < numsSize; j++)
        {
            if (nums[i] + nums[j] == target)
            {
                result[0] = i;
                result[1] = j;
                // *returnSize = 2;
                printf("%d\n", *returnSize);
            }
        }
    }
    return result;
}

int main()
{
    int nums[] = {2, 7, 11, 15};
    int target = 9;
    int returnSize = 2;
    int *result = twoSum(nums, 4, target, &returnSize);
    printf("%d %d\n", result[0], result[1]);
    return 0;
}
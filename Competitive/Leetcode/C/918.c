/**
 * @file 918.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Maximum sum circular array
 * @version 1.0
 * @date 2023-01-18
 *
 * @copyright Copyright (c) 2023
 *
 */

/*

Given a circular integer array nums of length n, return the maximum possible sum of a non-empty subarray of nums.

A circular array means the end of the array connects to the beginning of the array. Formally, the next element of nums[i] is nums[(i + 1) % n] and the previous element of nums[i] is nums[(i - 1 + n) % n].

A subarray may only include each element of the fixed buffer nums at most once. Formally, for a subarray nums[i], nums[i + 1], ..., nums[j], there does not exist i <= k1, k2 <= j with k1 % n == k2 % n.

 */

// #include <stdio.h>
// #include <stdlib.h>

// int maxSubarraySumCircular(int* nums, int numsSize){
//     int max = nums[0];
//     int min = nums[0];
//     int maxSum = nums[0];
//     int minSum = nums[0];
//     int total = nums[0];
//     for(int i = 1; i < numsSize; i++){
//         max = max > 0 ? max + nums[i] : nums[i];
//         min = min < 0 ? min + nums[i] : nums[i];
//         maxSum = maxSum > max ? maxSum : max;
//         minSum = minSum < min ? minSum : min;
//         total += nums[i];
//     }
//     return maxSum > 0 ? (total - minSum > maxSum ? total - minSum : maxSum) : maxSum;
// }

// int main(){
//     int nums[] = {-2,4,-5,4,-5,9,4};
//     int numsSize = sizeof(nums)/sizeof(nums[0]);
//     printf("%d\n", maxSubarraySumCircular(nums, numsSize));
//     return 0;
// }

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int maxSubarraySumCircular(int* nums, int numsSize){
    int max = nums[0];
    int maxSum = nums[0];
    int total = nums[0];
    int minSum = 0;
    for(int i = 1; i < numsSize; i++){
        max = fmax(max + nums[i], nums[i]);
        maxSum = fmax(maxSum, max);
        total += nums[i];
        minSum = fmin(minSum + nums[i], minSum);
    }
    return maxSum > 0 ? fmax(total - minSum, maxSum) : maxSum;
}

int main(){
    int nums[] = {-2,4,-5,4,-5,9,4};
    int numsSize = sizeof(nums)/sizeof(nums[0]);
    printf("%d\n", maxSubarraySumCircular(nums, numsSize));
    return 0;
}

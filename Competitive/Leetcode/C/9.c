/**
 * @file 2.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Palindrome checker
 * @version 1.0
 * @date 2022-11-28
 *
 * @copyright Copyright (c) 2022
 *
 */

// Given an integer x, return true if x is a palindrome, and false otherwise.

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

bool isPalindrome(int x)
{
    int temp = x;
    int rev = 0;
    while (temp > 0)
    {
        rev = rev * 10 + temp % 10;
        temp /= 10;
    }
    if (rev == x)
    {
        return true;
    }
    else
    {
        return false;
    }
}

int main()
{
    int x = 120;
    printf("%d\n", isPalindrome(x));
    return 0;
}
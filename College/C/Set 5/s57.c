/**
 * @file s57.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Palindrom
 * @version 1.0
 * @date 2022-05-19
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int n, r = 0, temp;
    printf("Enter a number to Check: ");
    scanf("%d", &n);

    temp = n;
    while (n != 0)
    {
        r *= 10;
        r += n % 10;
        n /= 10;
    }
    if (temp == r)
    {
        printf("It is a Palindrom!\n");
    }
    else
    {
        printf("It is not a Palindrom!\n");
    }

    return 0;
}
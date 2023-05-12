/**
 * @file s23.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Reversing a number.
 * @version 1.0
 * @date 2022-04-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
int main()
{
    int n, r = 0;

    printf("Enter a number to reverse\n");
    scanf("%d", &n);

    while (n != 0)
    {
        r *= 10;
        r += n % 10;
        n /= 10;
    }

    printf("Reverse of the number = %d\n", r);
    return 0;
}
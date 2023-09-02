/**
 * @file s52.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief First N odd sum
 * @version 1.0
 * @date 2022-04-28
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int start = 1, n, res = 0;

    printf("Total numbers: ");
    scanf("%d", &n);

    while (start <= n)
    {
        res += ((start * 2) - 1);
        start++;
    }
    printf("Final Result: %d\n", res);

    return 0;
}
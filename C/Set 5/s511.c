/**
 * @file s59.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief pattern
 * @version 1.0
 * @date 2022-05-19
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <stdio.h>

int main()
{

    int i, j, n = 5;
    for (i = 1; i <= n; i++)
    {
        for (j = 0; j < i; j++)
        {
            printf("* ");
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}
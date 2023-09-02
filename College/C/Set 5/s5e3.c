/**
 * @file s5e3.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Pattern 1 to 5
 * @version 1.0
 * @date 2022-05-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int i, j, n = 5;
    for (i = n; i > 0; i--)
    {
        for (j = n; j > i; j--)
        {
            printf("  ");
        }
        for (j = 0; j < i; j++)
        {
            printf(" %d", j + 1);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}
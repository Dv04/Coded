/**
 * @file s58.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief prime between 100
 * @version 1.0
 * @date 2022-05-19
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int n = 100, check = 1;
    for (int i = 1; i < n; i++)
    {
        check = 1;
        for (int j = 1; j <= (i / 2); j++)
        {
            if (i % j == 0)
            {
                check += 1;
            }
        }
        if (check == 2)
        {
            printf("%d, ", i);
        }
    }
    printf("\n");
    return 0;
}
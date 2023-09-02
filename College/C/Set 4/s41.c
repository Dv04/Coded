/**
 * @file s41.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Mazimum of three numbers
 * @version 1.0
 * @date 2022-04-22
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int a, b, c;

    printf("Number 1: ");
    scanf("%d", &a);
    printf("Number 2: ");
    scanf("%d", &b);
    printf("Number 3: ");
    scanf("%d", &c);

    // if (a<b)
    // {
    //     if (b<c)
    //     {
    //         printf("Number 3 is the largest number\n");
    //     }
    //     else
    //     {
    //         printf("Number 2 is the largest number\n");
    //     }

    // }
    // else
    // {
    //     if (a<c)
    //     {
    //         printf("Number 3 is the largest number\n");
    //     }
    //     else
    //     {
    //         printf("Number 1 is the largest number\n");
    //     }

    // }

    int ans = (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
    printf("%d\n", ans);

    return 0;
}
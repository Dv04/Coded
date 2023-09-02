/**
 * @file s31.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Maximum from given two nos.
 * @version 1.0
 * @date 2022-04-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int a, b;

    printf("Enter the first number: ");
    scanf("%d", &a);
    printf("Enter the second number: ");
    scanf("%d", &b);

    int ans = (a > b) ? a : b;

    printf("The bigger number is %d\n", ans);

    return 0;
}
/**
 * @file Untitled-1
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Divisibility
 * @version 1.0
 * @date 2022-04-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int x, y;
    printf("Enter first number: ");
    scanf("%d", &x);
    printf("Enter Second number: ");
    scanf("%d", &y);

    if (x % y == 0)
        printf("X is divisible by y\n");
    else
        printf("X is not divisible by y\n");

    return 0;
}
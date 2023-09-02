/**
 * @file Decision.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Decision making
 * @version 1.0
 * @date 2022-04-26
 *
 * @copyright Copyright (c) 2022
 *
 */

/* Decision making statements allow you to decide the order of execution of specific statements.
 Common decisions making are as follows.
 1. If-else
 2. Loop structures
 3. nested structures
 4. switch statements
 */

#include <stdio.h>

int main()
{
    // int count = 1, n = 5, x, sum = 0;
    // while (count < n)
    // {
    //     printf("%d\n", count);
    //     scanf("%d", &x);
    //     count++;
    //     sum += x;
    // }
    // printf("%d\n", sum);

    int x, y;
    printf("Enter first number: ");
    scanf("%d", &x);
    printf("Enter Second number: ");
    scanf("%d", &y);

    if(x%y==0)
        printf("X is divisible by y\n");
    else
        printf("X is not divisible by y\n");
    
    return 0;
}
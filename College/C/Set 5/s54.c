/**
 * @file s54.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief factorials
 * @version 1.0
 * @date 2022-04-28
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int n, fact = 1;

    printf("Enter the number: ");
    scanf("%d", &n);

    while (n > 0)
    {
        fact *= n;
        // printf("%d\n", fact);
        n--;
    }
    printf("%d\n", fact);
    return 0;
}
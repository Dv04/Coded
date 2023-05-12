/**
 * @file s13.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Change two variables with third variable.
 * @version 1.0
 * @date 2022-04-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int a, b, c;

    printf("Enter first number: ");
    scanf("%d", &a);
    printf("Enter second number: ");
    scanf("%d", &b);

    c = a;
    a = b;
    b = c;

    printf("The changed numbers are:\n a = %d\n b = %d\n", a, b);

    return 0;
}
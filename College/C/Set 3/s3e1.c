/**
 * @file s3e1.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief data size
 * @version 1.0
 * @date 2022-04-28
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{
    int x, y;
    float z;
    double a, b;
    char c;
    int x2 = sizeof(x), y2 = sizeof(y), z2 = sizeof(z), a2 = sizeof(a), b2 = sizeof(b), c2 = sizeof(c);
    printf("The size of x is %d\nThe size of y is %d\nThe size of z is %d\nThe size of a is %d\nThe size of b is %d\nThe size of c is %d\n", x2, y2, z2, a2, b2, c2);

    // int res = x2 + a2 + b2 + c2 + y2 + z2;

    // printf("%d\n", res);

    return 0;
}
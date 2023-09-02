/**
 * @file s3e2.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief operators
 * @version 1.0
 * @date 2022-05-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{
    int X, Y, Z, T;
    Y = 6;
    X = Y--;
    X++;
    X = ++Y;
    T = Z++ + ++Y;
    T += 8;
    T += (Z++);
    T += (++Z);

    printf("The value of X is %d\n", X);
    printf("The value of Y is %d\n", Y);
    printf("The value of Z is %d\n", Z);
    printf("The value of T is %d\n", T);

    return 0;
}
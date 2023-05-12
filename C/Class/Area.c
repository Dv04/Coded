/**
 * @file Eg.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Area of rectangle
 * @version 1.0
 * @date 2022-04-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#define PI 3.14

int main()
{

    int l = 4, b = 5, r = 45;

    int peri = 2 * (l + b), arear = l * b;
    float areac = PI * r * r;

    printf("%d %d %.3f\n", peri, arear, areac);
    if (peri != (areac / 4))
    {
        printf("This condition is not valid\n");
    }

    return 0;
}
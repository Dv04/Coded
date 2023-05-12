/**
 * @file s2e1.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Area of a circle with given radius.
 * @version 1.0
 * @date 2022-04-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#define PI 3.14

int main()
{

    int r;
    printf("Enter the Radius: ");
    scanf("%d", &r);

    float area = PI * (r * r);

    printf("%.3f\n", area);

    return 0;
}
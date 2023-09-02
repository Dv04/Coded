/**
 * @file s9e1.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief power function
 * @version 1.0
 * @date 2022-05-31
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <math.h>

double power(double x, double y)
{
    return pow(x, y);
}

int main()
{

    int x, y;
    printf("Enter the number to power: ");
    scanf("%d", &x);
    printf("Enter the power factor: ");
    scanf("%d", &y);
    printf("%.3lf\n", power(x, y));
    return 0;
}
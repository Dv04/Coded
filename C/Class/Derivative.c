/**
 * @file Derivative.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Derivatives
 * @version 1.0
 * @date 25-04-2022
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <math.h>

int main()
{

    float x;
    printf("Enter the value of x: ");
    scanf("%f", &x);

    // float fx = (exp(-pow(x,3))*cos(x));
    float gx = -3 * pow(x, 2) * exp(-pow(x, 3)) * cos(x) - sin(x) * exp(-pow(x, 3));

    printf("The derivative of f(x) at x = %f is %.3f.\n", x, gx);

    return 0;
}

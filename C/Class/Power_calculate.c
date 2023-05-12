/**
 * @file Untitled-1
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Power
 * @version 1.0
 * @date 2022-04-25
 * @copyright Copyright (c) 2022
 */

#include <stdio.h>
#include <math.h>
#define pi 22 / 7

int main()
{

    double current, volt = 220, Z, P, Q, S, angle;

    Z = sqrt(3 * 3 + 4 * 4);
    angle = (atan(4 / 3)) * 180 / pi;
    current = volt / Z;
    P = volt * current * cos(angle);
    Q = volt * current * sin(angle);
    S = sqrt(P * P + Q * Q);

    // printf("%f\n", cos(angle));
    printf("The value of current is : %.4f\n", current);
    printf("The value of Real power is : %.4f\n", P);
    printf("The value of Imagiary power is : %.4f\n", Q);
    printf("The value of Total power is : %.4f\n", S);

    return 0;
}
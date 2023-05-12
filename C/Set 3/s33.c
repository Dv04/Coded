/**
 * @file s33.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Simple interest
 * @version 1.0
 * @date 2022-04-28
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int p, n;
    float r, i;

    printf("Enter p, r, n: ");
    scanf("%d %f %d", &p, &r, &n);
    i = (float)(p * r * n) / 100;

    printf("%.3f\n", i);

    return 0;
}
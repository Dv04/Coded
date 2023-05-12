/**
 * @file Array.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Arrays
 * @version 1.0
 * @date 2022-05-10, 2022-05-11
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#define NL \n
#define EOL '\0'

int main()
{

    // 1. int a[10][10][10];
    // for (int i = 0; i < 2; i++)
    // {
    //     for (int j = 0; j < 2; j++)
    //     {
    //         for (int k = 0; k < 2; k++)
    //         {
    //             scanf("%d", &a[i][j][k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("---\n\n");
    // }
    // for (int i = 0; i < 2; i++)
    // {
    //     for (int j = 0; j < 2; j++)
    //     {
    //         for (int k = 0; k < 2; k++)
    //         {
    //             printf("%d\n", a[i][j][k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // 2.  static float x[10];
    // for (int i = 4; i < 10; i++)
    // {
    //     scanf("%f", &x[i]);
    // }
    // for (int i = 1; i < 10; i++)
    // {
    //     printf("%.3f\n", x[i]);
    // }
    // 3. int i = 2;
    // for (; i < 10;)
    // {
    //     printf("%d\n", i);
    //     i++;
    // }
    // In C compilers, elements which are not initialised explicitely are initialized as 0, which is known as Zero padding.
    // 4. int a = 0, b = 0, c[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
    // for (; b < 10; ++b)
    // {
    //     a += c[b];
    // }
    // printf("%d\n", a);

    int size;
    // printf("Enter number of data input: ");
    // scanf("%d", &size);
    size = 10;
    // for (int i = 0; i < size; i++)
    // {
    //     scanf("%f",&x[i]);
    // }
    float mean, std, sum = 0.0, sum2 = 0.0, x[10] = {12.3, 11.4, 56.3, 112.4, 19.05, 11.31, 4.5, 22.91, 17.47, 9.55};
    for (int i = 0; i < 10; i++)
    {
        sum += x[i];
    }
    mean = sum / 10;
    printf("%.3f\n", mean);
    for (int i = 0; i < 10; i++)
    {
        sum2 += pow((x[i] - mean), 2);
        printf("%.3f\n", sum2);
    }
    std = sqrt(sum2 / 9);
    printf("Standard deviation is: %.3f\n", std);

    // homework matrix normalization code. 2x2 3x3 and 10x13

    return 0;
}
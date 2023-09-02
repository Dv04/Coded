/**
 * @file Untitled-1
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief
 * @version 1.0
 * @date 2022-06-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <math.h>

// int power(float velocity[], float area, float *p[])
// {
//     // int p[10];
//     for (int i = 0; i < 10; i++)
//     {

//         *p[i] = (0.5 * area * pow(velocity[i], 3));
//         // printf("%f \n",0.5*area*pow(velocity[i],3));
//         // printf("p=%.3f\n",*p[i]);
//     }
//     return *p[10];
// }
// int main()
// {
//     int diameter = 120;
//     float area, p[10], velocity[] = {12, 10.5, 11.4, 4.5, 6.32, 7.31, 8.10, 10.01, 2.41, 4.5};

//     area = (M_PI * pow(diameter, 2) * 0.25);
//     printf("%f\n", area);
//     printf("The array of power is :");
//     power(velocity, area, &p);
//     for (int i = 0; i < 10; i++)
//     {
//         printf("%f\n", p[i]);
//     }

//     return 0;
// }

// #include <stdio.h>
// int *getarray(int *a)
// {

//     printf("Enter the elements in an array : ");
//     for (int i = 0; i < 5; i++)
//     {
//         scanf("%d", &a[i]);
//     }
//     return a;
// }
// int main()
// {
//     int *n;
//     int a[5];
//     n = getarray(a);
//     printf("\nElements of array are :");
//     for (int i = 0; i < 5; i++)
//     {
//         printf("%d\n", n[i]);
//     }
//     return 0;
// }

int main()
{
    double h = 6.626 / pow(10, 34);
    printf("%.40lf\n", h);
    return 0;
}
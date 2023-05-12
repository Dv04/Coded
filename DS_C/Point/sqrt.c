/**
 * @file sqrt.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief write a program to calculate square of a number using call by value.
 * @version 1.0
 * @date 2022-09-26
 *
 * @copyright Copyright (c) 2022
 *
 */

/*
    The logic is to use a function to calculate the square of a number using call by reference.
    we use a function to calculate the square of a number and then we print the value of the square.
*/

#include <stdio.h>

void sqr(float *a)
{
    *a = *a * *a;
}

int main()
{

    float a;
    printf("Enter the number: ");
    scanf("%f", &a);
    sqr(&a);
    printf("The square of the number is %.4f\n", a);
    printf("\n");

    return 0;
}
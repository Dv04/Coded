/**
 * @file s9e4.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Calculator
 * @version 1.0
 * @date 2022-05-31
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int Add(int a, int b)
{
    return printf("%d\n\n", a + b);
}
int Sub(int a, int b)
{
    return printf("%d\n\n", (a - b));
}
int Mul(int a, int b)
{
    return printf("%d\n\n", a * b);
}
float Div(float a, float b)
{
    return printf("%.3f\n\n", a / b);
}

int main()
{

    printf("Welcome to Calculator!\n\n");

    int a, b;
    printf("Enter first number: ");
    scanf("%d", &a);
    printf("Enter second number: ");
    scanf("%d", &b);

    int cal;
    printf("\n\nPress 1 for Addition\nPress 2 for Subtraction\nPress 3 for Multiplication\nPress 4 for Division\n\nEnter the number: ");
    scanf("%d", &cal);

    switch (cal)
    {
    case 1:
        Add(a, b);
        break;
    case 2:
        Sub(a, b);
        break;
    case 3:
        Mul(a, b);
        break;
    case 4:
        Div(a, b);
        break;
    default:
        printf("Invalid\n");
        break;
    }

    return 0;
}
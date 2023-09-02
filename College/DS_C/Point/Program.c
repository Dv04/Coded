/**
 * @file Program.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Write a program to perform some operations on variable using call by value and call by reference.
 * @version 1.0
 * @date 2022-09-26
 *
 * @copyright Copyright (c) 2022
 *
 */

/*
    The logic is to add or substract or multiply or divide the two floating point integers using reference pointers and a switch case.
    The user will be asked to choose which action it wants to perform and then the user will be asked to enter the two floating point numbers.

*/

#include <stdio.h>

void operator(int *x, int *y, int choice)
{
    float a = (float)*x;
    float b = (float)*y;

    switch (choice)
    {
    case 1:
        printf("The sum of the numbers is %d\n", *x + *y);
        break;
    case 2:
        printf("The difference of the numbers is %d\n", *x - *y);
        break;
    case 3:
        printf("The product of the numbers is %d\n", *x * *y);
        break;
    case 4:
        printf("The division of the numbers is %f\n", a / b);
        break;
    default:
        printf("Invalid choice");
        break;
    }
}

int main()
{

    int x, y;
    printf("Enter the first number: ");
    scanf("%d", &x);
    printf("Enter the second number: ");
    scanf("%d", &y);

    int choice;

    printf("Enter 1 if you want to add\nEnter 2 if you want to subtract\nEnter 3 if you want to multiply\nEnter 4 if you want to divide\n\n");
    scanf("%d", &choice);

    operator(&x, &y, choice);

    return 0;
}
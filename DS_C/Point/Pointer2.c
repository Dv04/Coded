/**
 * @file Pointer2.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Pointer by call by value.
 * @version 1.0
 * @date 2022-09-26
 * 
 * @copyright Copyright (c) 2022
 * 
 */

/*
    The logic is to create an integer temp which is used in user defined function to swap the values of both a and b and the result is printed.
*/

#include <stdio.h>


void swap1(int a, int b)
{
    int temp = a;
    a = b;
    b = temp;
    printf(" the new value of a is %d and b is %d\n", a, b);
}

int main()
{
    int a = 5, b = 10;
    printf("The value of a is %d and b is %d\n", a, b);
    swap1(a, b);
}
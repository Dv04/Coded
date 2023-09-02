/**
 * @file 1.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Write a program to swap values of two variables using call by reference. 
 * @version 1.0
 * @date 2022-09-26
 *
 * @copyright Copyright (c) 2022
 *
 */

/*
    The logic is to use a temporary variable to swap the values of two variables using call by reference.
    we use a temporary variable to store the value of a and then we assign the value of b to a and then we assign the value of temp to b.
*/

#include <stdio.h>
#include <string.h>

void swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main()
{
    int a = 5, b = 10;
    swap(&a, &b);
    printf("The value of a is %d and b is %d\n", a, b);
}
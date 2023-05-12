/**
 * @file Homework.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Homework for today
 * @version 1.0
 * @date 2022-04-27
 *
 * @copyright Copyright (c) 2022
 *
 */

/* Write a c code to determine if the user defined integer is a multiple of 3 or 5 or both
    write a c code to check if sun if roots of a quad eq x2 -4x + 8 is equal to product of roots
*/
#include <stdio.h>

int main()
{
    // {
    // int x;
    // printf("enter a number: ");
    // scanf("%d", &x);
    // if (x % 3 == 0 || x % 5 == 0)
    //     printf("It is a multiple of 3 or 5\n");
    // else
    //     printf("It is a multiple of none\n");
    // }

    int a = 1, b = -4, c = 8;

    if ((-b / a) == (c / a))
    {
        printf("It is\n");
    }
    else
    {
        printf("Its not\n");
    }

    return 0;
}


// How many days are present in a particular month, condition with entering  first three character of month
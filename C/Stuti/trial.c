/**
 * @file start.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Introduction for stuti
 * @version 1.0
 * @date 2022-05-07
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    /*
    Now if we want to print numbers 1 to 100 what will you do?

    Good you answered correctly.

    we have three loops, first one is:
    */

    int i = 1; // our starting integer.
    while (1)  // we have to run until I is less than 100.
    {
        if (i == 1000)
        {
            break; // code breaks down crumbling, shattered into a mysterious space.
        }
        printf("%d\n", i); // print the number.
        i++;               // increment i by 1.
    }

    return 0;
}
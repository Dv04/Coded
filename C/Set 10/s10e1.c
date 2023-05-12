/**
 * @file s10e1.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief 
 * @version 1.0
 * @date 2022-06-16
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>
int main()
{
    int x[10] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}, *ptr;
    ptr = &x[0];

    for (int i = 0; i < 10; i++)
    {
        printf("\n%d element of array is :%d\n", i + 1, x[i]);
        printf("The address of %d element is :%u\n", i + 1, (ptr + i));
    }

    return 0;
}


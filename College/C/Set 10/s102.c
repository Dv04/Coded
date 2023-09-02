/**
 * @file s102.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief N different integer numbers and calculate their sum using pointer.
 * @version 1.0
 * @date 2022-06-16
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int main()
{
    int *ptr1, *ptr2, num, sum;
    ptr1 = &sum;
    ptr2 = &num;
    *ptr1 = 0;

    for (int i = 1; i <= 5; i++)
    {
        printf("Enter the number: ");
        scanf("%d", ptr2);
        *ptr1 += *ptr2;
    }
    printf("The sum is : %d\n", *ptr1);
    return 0;
}


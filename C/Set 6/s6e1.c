/**
 * @file s6e1.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Largest of given numbers
 * @version 1.0
 * @date 2022-05-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int main()
{
    int n = 10,temp;

    int arr[n];
    for (int i = 0; i < n; i++)
    {
        printf("Enter a number: ");
        scanf("%d", &arr[i]);
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            if (arr[j] > arr[i])
            {
                temp = arr[j];
                arr[j] = arr[i];
                arr[i] = temp;
            }
        }
    }
    printf("\n");

    printf("The biggest number from given input is: %d\n", arr[0]);

    return 0;
}
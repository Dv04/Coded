/**
 * @file s55.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief second largest number
 * @version 1.0
 * @date 2022-05-05
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int main()
{
    int n,temp;
    printf("How many numbers do you want to input: ");
    scanf("%d", &n);

    int arr[n];
    for (int i = 0; i < n; i++)
    {
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
    for (int i = 1; i < n; i++)
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

    // for (int i = 0; i < n; i++)
    // {
        printf("The second biggest number from given input is: %d\n", arr[1]);
    // }
    return 0;
}
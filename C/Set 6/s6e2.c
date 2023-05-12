/**
 * @file s6e2.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Read an array and print it in reverse
 * @version 1.0
 * @date 2022-05-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int n;
    printf("How many number: ");
    scanf("%d", &n);

    int arr[n];
    for (int i = 1; i <= n; i++)
    {
        printf("Enter a number: ");
        scanf("%d", &arr[n - i]);
    }
    printf("\n");
    for (int i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
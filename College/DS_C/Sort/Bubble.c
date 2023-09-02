/**
 * @file Bubble.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Bubble sort function
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int bubble_sort(int arr[], int n)
{
    int i, j, temp;
    int num = 0;
    int change = 0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                change = 1;
            }
        }
        num += 1;
        printf("Number of passes: %d\n", num);
        if (change == 0)
        {
            printf("Sorted.\n");
            break;
        }
    }
    printf("Sorted.\n");
    return 0;
}

int main()
{

    int arr[10] = {1, 2, 3, 24, 5, 61, 7, 8, 9, 10};

    bubble_sort(arr, 10);

    for (int i = 0; i < 10; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
    return 0;
}
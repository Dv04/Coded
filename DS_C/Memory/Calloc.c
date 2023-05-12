/**
 * @file Malloc.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief write a program to assign memory location using calloc and malloc to array of n elements.
 * @version 1.0
 * @date 2022-09-26
 *
 * @copyright Copyright (c) 2022
 *
 */

/*
    The logic:
    1. Take the size of array from user.
    2. Assign memory location to array using malloc and calloc.
    3. Check if memory is assigned or not.
    4. If memory is not allocated then print error message.
    5. If memory is allocated then scan the array using for loop.
    6. Print the array using for loop with the addresses of each element.
    7. Free the memory location using free function.
    8. Print the free memory message.
    9. Exit.
*/

#include <stdio.h>
#include <stdlib.h>

int main()
{

    int *arr, n, *arr1;
    printf("Enter the size of the array: ");
    scanf("%d", &n);

    arr = (int *)calloc(n, sizeof(int));

    arr1 = (int *)malloc(n * sizeof(int));

    if (arr == NULL || arr1 == NULL)
    {
        printf("Memory allocation failed\n");
        exit(1);
    }

    else
    {
        printf("\nMemory allocated successfully and the size of array is %ld and array1 is %ld \n", sizeof(arr), sizeof(arr1));
        printf("\n");

        for (int i = 0; i < n; i++)
        {
            printf("Enter the value of element %d for array 1: ", i + 1);
            scanf("%d", &arr[i]);
        }
        printf("\n");

        for (int i = 0; i < n; i++)
        {
            printf("Enter the value of element %d for array 2 : ", i + 1);
            scanf("%d", &arr1[i]);
        }

        printf("\n");

        for (int i = 0; i < n; i++)
        {
            printf("The value of element %d for array 1 is %d with address of: %p", i + 1, arr[i], &arr[i]);
            printf("\n");
        }

        printf("\n");

        for (int i = 0; i < n; i++)
        {
            printf("The value of element %d for array 2 is %d with address of: %p", i + 1, arr1[i], &arr1[i]);
            printf("\n");
        }
        }

    printf("\n");

    free(arr);
    free(arr1);

    printf("Memory freed successfully\n");

    return 0;
}
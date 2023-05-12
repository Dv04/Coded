/**
 * @file Untitled-1
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Write a program to perform dynamic memory allocation using calloc, also for the new array size use realloc function.
 * @version 1.0
 * @date 2022-10-03
 *
 * @copyright Copyright (c) 2022
 *
 */

/*
    Logic:
    1. Take the size of array from user.
    2. Assign memory location to array using calloc.
    3. Check if memory is assigned or not.
    4. If memory is not allocated then print error message.
    5. If memory is allocated then scan the array using for loop.
    6. Print the array using for loop with the addresses of each element.
    7. Take new size of array from user.
    8. Reallocate the memory location using realloc function.
    9. Enter new values in the array.
    10. Print the array using for loop with the addresses of each element.
    11. Free the memory location using free function.
    12. Print the free memory message.
    13. Exit.
*/

#include <stdio.h>
#include <stdlib.h>

int main()
{

    int *arr, n, temp;
    printf("Enter the size of the array: ");
    scanf("%d", &n);
    temp = n;

    arr = (int *)calloc(n, sizeof(int));

    if (arr == NULL)
    {
        printf("Memory allocation failed\n");
        exit(1);
    }

    else
    {
        printf("\nMemory allocated successfully and the size of array is %ld \n", sizeof(arr));
        printf("\n");

        for (int i = 0; i < n; i++)
        {
            printf("Enter the value of element %d for array 1: ", i + 1);
            scanf("%d", &arr[i]);
        }
        printf("\n");

        for (int i = 0; i < n; i++)
        {
            printf("The value of element %d for array 1 is %d and the address is %p", i + 1, arr[i], &arr[i]);
            printf("\n");
        }
        printf("\n");

        printf("Enter the new size of the array: ");
        scanf("%d", &n);

        arr = realloc(arr, n * sizeof(int));

        if (temp > n)
        {
            for (int i = 0; i < n; i++)
            {
                printf("Enter the value of element %d for array 2 : ", i + 1);
                scanf("%d", &arr[i]);
            }
            printf("\n");

            for (int i = 0; i < n; i++)
            {
                printf("The value of element %d for array 1 is %d and the address is %p", i + 1, arr[i], &arr[i]);
                printf("\n");
            }
        }
        else
        {
            for (int i = temp; i < n; i++)
            {
                printf("Enter the value of element %d for array 2 : ", i + 1);
                scanf("%d", &arr[i]);
            }
            printf("\n");

            for (int i = 0; i < n; i++)
            {
                printf("The value of element %d for array 1 is %d and the address is %p", i + 1, arr[i], &arr[i]);
                printf("\n");
            }
        }
    }

    printf("\n");

    free(arr);
    printf("Memory freed successfully\n");

    return 0;
}
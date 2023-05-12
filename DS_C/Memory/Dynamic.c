/**
 * @file Dynamic.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Write a program to perform memory allocation for an array of integers using malloc() and free() functions.
 * @version 1.0
 * @date 2022-10-03
 *
 * @copyright Copyright (c) 2022
 *
 */

/*
    The logic is to use malloc() to allocate memory for an array of integers and then use free() to free the allocated memory.
    malloc allocates memory with size of intergers and N number of elements. then we add the value using scanf() and then print the value using printf().
    and finally it is printed.
 */

#include <stdio.h>
#include <stdlib.h>

int main()
{

    int *arr, n;
    printf("Enter the size of the array: ");
    scanf("%d", &n);
    arr = (int *)malloc(n * sizeof(int));

    if (arr == NULL)
    {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < n; i++)
    {
        printf("Enter the value of %d element: ", i);
        scanf("%d", &arr[i]);
    }
    printf("\n");
    
    for (int i = 0; i < n; i++)
    {
        printf("The value of %d element is %d at %p", i, arr[i], &arr[i]);
        printf("\n");
    }

    free(arr);

    return 0;
}
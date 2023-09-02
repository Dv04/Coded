/**
 * @file Insertion.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Insertion sort function
 * @version 1.0
 * @date 2022-12-26
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int main(){

    int arr[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    int i, j, temp;

    for (i = 1; i < 10; i++)
    {
        temp = arr[i];
        j = i - 1;
        while (j >= 0 && arr[j] > temp)
        {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = temp;
    }

    for (i = 0; i < 10; i++)
    {
        printf("%d ", arr[i]);
    }

    printf("\n");

    return 0;
}
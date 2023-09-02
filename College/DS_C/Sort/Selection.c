/**
 * @file Selection.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Selection sort 
 * @version 1.0
 * @date 2022-12-26
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int Selection_sort( int arr[], int n ){
    int i, j, temp, min;
    for ( i = 0; i < n; i++ ){
        min = i;
        for ( j = i + 1; j < n; j++ ){
            if ( arr[j] < arr[min] ){
                min = j;
            }
        }
        temp = arr[i];
        arr[i] = arr[min];
        arr[min] = temp;
    }
    for ( i = 0; i < n; i++ ){
        printf("%d ", arr[i]);
    }
    return 0;
}

int main(){

    int arr[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    Selection_sort(arr, 10);

    printf("\n");

    return 0;
}
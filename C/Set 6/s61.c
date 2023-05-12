/**
 * @file s61.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Sum and average of 10 elements of array.
 * @version 1.0
 * @date 2022-05-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int arr[10], i = 0, sum = 0;
    float avg;
    while (i<10)
    {
        printf("Enter an element: ");
        scanf("%d", &arr[i]);
        sum+= arr[i];
        i++;
    }

    avg = (float)sum / 10;

    printf("The sum is: %d\nThe average is: %.3f\n", sum, avg);

    return 0;
}
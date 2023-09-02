/**
 * @file start.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Introduction for stuti
 * @version 1.0
 * @date 2022-05-07
 *
 * @copyright Copyright (c) 2022
 *
 */

// binary search
#include <stdio.h>
int main()
{
    int a[9], n = 9, mid, low = 0, high = n - 1, i, val, temp;
    printf("\nenter array elements in sorted:\n");
    for (i = 0; i < n; i++)
    {
        printf("enter value: ");
        scanf("%d", &a[i]);
    }
    printf("\nenter search element: ");
    scanf("%d", &val);

    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            if (a[j] < a[i])
            {
                temp = a[j];
                a[j] = a[i];
                a[i] = temp;
            }
        }
    }
    while (low <= high)
    {
        mid = (int)((low + high) / 2);
        // printf("2 %d %d\n", a[mid], mid);
        if (a[mid] == val)
        {
            printf("\nvalue found on %d position\n", mid + 1);
            break;
        }
        else if (a[mid] < val)
        {
            low = mid + 1;
            // printf("A - %d %d\n", low, high);
        }
        else if (a[mid] > val)
        {
            high = mid - 1;
            // printf("B - %d %d\n", low, high);
        }
        else
        {
            printf("\nvalue not found\n");
        }
    }
    return 0;
}
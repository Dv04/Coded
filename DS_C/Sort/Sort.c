/**
 * @file Sort.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief All sort functions
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <stdlib.h>

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
        if (change == 0)
        {
            printf("Number of passes: %d\n", num);
            printf("Sorted.\n");
            return 0;
        }
    }
    printf("Number of passes: %d\n", num);
    printf("Sorted.\n");
    return 0;
}

int Selection_sort(int arr[], int n)
{
    int i, j, temp, min;
    for (i = 0; i < n; i++)
    {
        min = i;
        for (j = i + 1; j < n; j++)
        {
            if (arr[j] < arr[min])
            {
                min = j;
            }
        }
        temp = arr[i];
        arr[i] = arr[min];
        arr[min] = temp;
    }
    for (i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
    }
    return 0;
}

int Insertion_sort(int arr[], int n)
{
    int i, j, temp;
    for (i = 1; i < n; i++)
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
    for (i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
    }
    return 0;
}

int Merge_sort(int arr[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
    return 0;
}

void mergeSort(int arr[], int l, int r)
{
    if (l < r)
    {

        int m = l + (r - l) / 2;

        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        Merge_sort(arr, l, m, r);
    }
}

int partition(int arr[], int l, int r)
{
    int pivot = arr[r];
    int i = (l - 1);

    for (int j = l; j <= r - 1; j++)
    {
        if (arr[j] < pivot)
        {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    int temp = arr[i + 1];
    arr[i + 1] = arr[r];
    arr[r] = temp;
    return (i + 1);
}

int Quick_sort(int arr[], int l, int r)
{
    if (l < r)
    {
        int pi = partition(arr, l, r);

        Quick_sort(arr, l, pi - 1);
        Quick_sort(arr, pi + 1, r);
    }
    return 0;
}

int print(int arr[], int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
    }
    return 0;
}

int main()
{

    int choice;
    int size;
    int arrsize;
    goto changing;
changing:
    printf("\n\nEnter your size: ");
    scanf("%d", &size);
    int arr[size];
    for (int i = 0; i < size; i++)
    {
        printf("Enter your number: ");
        scanf("%d", &arr[i]);
    }

    printf("\n\nWhich sort do you want to use?\nPress 1 for Bubble sort\nPress 2 for Selection sort\nPress 3 for Insertion sort\nPress 4 for Merge sort\nPress 5 for Quick sort\nPress 6 to exit\n");

    do
    {

        printf("\nEnter your choice: ");
        scanf("%d", &choice);
        printf("\n");
        switch (choice)
        {
        case 1:
            printf("Using Bubble Sort\n");
            bubble_sort(arr, size);
            print(arr, size);
            break;
        case 2:
            printf("Using Selection Sort\n");
            Selection_sort(arr, size);

            break;
        case 3:
            printf("Using Insertion Sort\n");
            Insertion_sort(arr, size);

            break;
        case 4:
            printf("Using Merge Sort\n");
            mergeSort(arr, 0, size - 1);
            print(arr, size);
            break;
        case 5:
            printf("Using Quick Sort\n");
            Quick_sort(arr, 0, size - 1);
            print(arr, size);
            break;
        case 6:
            printf("Exiting...\n");
            exit(0);
            break;
        default:
            printf("Invalid choice.\n");
            break;
        }
        printf("\n\n");
        printf("Do you want to change the size of the array?\nPress 1 for yes\nPress 2 for no\nwhat is your choice: ");
        scanf("%d", &arrsize);
        if (arrsize == 1)
        {
            goto changing;
        }
    } while (choice != 6);
    return 0;
}
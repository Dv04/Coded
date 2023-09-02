/**
 * @file pointers.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Pointers
 * @version 1.0
 * @date 2022-06-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{
    static long int arr[10] = {' ', 1, 2, 3, 4, 5, 6, 7, 8, 9};
    char  *ptr[] = {"Ranbir", "Hello"};
    int a = 1;
    int *p1 = &a;
    
    // int **p2 = &p1;
    // ptr[0] = &arr[0]; 
    // printf("%d %d %d %d %d %d\n", a, p1, p2, *&p2, &*p2, &p2);
    
    printf("%d\n", *p1);
    p1++;
    printf("%d\n", *p1);
    
    return 0;
}
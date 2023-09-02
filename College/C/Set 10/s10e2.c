/**
 * @file s10e2.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief 
 * @version 1.0
 * @date 2022-06-16
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int main()
{
    int x, y;

    FILE *fptr;
    fptr = fopen("sum.txt", "r");
    fscanf(fptr, "%d", &x);
    fscanf(fptr, "%d", &y);
    fclose(fptr);

    fptr = fopen("answer.txt", "w");
    fprintf(fptr, "The sum is :%d", x + y);
    fclose(fptr);

    return 0;
}

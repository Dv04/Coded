/**
 * @file s104.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief read contents from one file and write into contents into another file
 * @version 1.0
 * @date 2022-06-16
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <stdio.h>

int main()
{
    char x;

    FILE *fptr,*fptr2;
    
    fptr = fopen("input.txt", "r");
    fptr2 = fopen("output.txt", "w");
    x = fgetc(fptr);
    while (x != EOF)
    {
        fputc(x, fptr2);
        x = fgetc(fptr);
    }
    fclose(fptr);
    fclose(fptr2);
    return 0;
}

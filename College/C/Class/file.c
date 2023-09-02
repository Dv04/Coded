/**
 * @file file.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief File management
 * @version 1.0
 * @date 2022-06-29
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    FILE *fp = fopen("file.txt", "r");

    char a[10];

    fgets(a, 5, fp);
    printf("%s\n", a);

    // char a = fgetc(fp);
    //  printf("%c\n", a);

    fclose(fp);

    return 0;
}

/**
 * creating a new file with fopen function with a/a+/w/w++
 * opening an existing file with fopen read mode.
 * reading from a file with fscanf or fgets
 * writing in a file with fprintf or fputs
 * moving to a specific position with fseek or rewing
 * closing a file with fclose
 */
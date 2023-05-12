/**
 * @file s71.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Half of a given string
 * @version 1.0
 * @date 2022-05-31
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <string.h>

int main()
{
    char str[] = "Dev_Sanghvi";

    printf("The string is :");
    for (int i = 0; i < (strlen(str) / 2); i++)
    {
        printf("%c", str[i]);
    }
    printf("\n");
    return 0;
}
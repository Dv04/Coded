/**
 * @file s74.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief convert as string into toggle string
 * @version 1.0
 * @date 2022-06-16
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>

int main()
{
    char str[] = "Dev", result[10];
    printf("The original string is : %s\n", str);
    printf("The toggled string is : ");
    for (int i = 0; i < strlen(str); i++)
    {
        if (isupper(str[i]))
        {
            result[i] = tolower(str[i]);
        }
        else if (islower(str[i]))
        {
            result[i] = toupper(str[i]);
        }

        printf("%c", result[i]);
    }
    printf("\n");
    return 0;
}
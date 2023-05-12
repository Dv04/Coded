/**
 * @file s7e2.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief count total words in a text.
 * @version 1.0
 * @date 2022-06-16
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>
#include <string.h>

int main()
{
    char str[] = "My name is-Dev Sanghvi";
    int k = 0, g = 0;
    for (int i = 0; i < strlen(str); i++)
    {
        if (str[i] == ' ')
        {
            k++;
        }
        else
        {
            continue;
        }
    }
    for (int i = 0; i < strlen(str); i++)
    {
        if (str[i] == ' ' || str[i] == '-' || str[i] == '_')
        {
            continue;
        }
        else
        {
            g++;
        }
    }
    printf("The total words in the string \"%s\" are :%d\n", str, k + 1);
    printf("The total characters in the string \"%s\" are :%d\n", str, g);
    return 0;
}
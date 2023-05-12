/**
 * @file s73.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief  find given string is palindrome or not.
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
    char str[] = "malayalam";
    static int k,g;
    char str1[30];
    int i;

    // Reverse string
    for (i = 0; i < strlen(str); i++)
    {
        str1[strlen(str) - (i + 1)] = str[i];
    }

    for (int i = 0; i < strlen(str); i++)
    {
        if (str1[i] == str[i])
        {
            k++;
        }
        else
        {
            g++;
        }
        
    }
    
    if (g == 0)
    {
        printf("The string \" %s \" is a palindrome.\n", str);
    }
    else
    {
        printf("The string %s is not a palindrome.\n", str);
    }

    return 0;
}
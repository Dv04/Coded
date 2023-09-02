/**
 * @file s72.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief copy one string into another string. Note:- 1) with using
strcpy( ) 2) without using strcpy( )
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
    char str1[] = "Malayalam", str2[10], str3[10];
    strcpy(str2, str1);
    printf("%s\n", str2);

    for (int i = 0; i < strlen(str1); i++)
    {
        str3[i] = str1[i];
    }
    printf("%s\n", str3);

    return 0;
}
/**
 * @file s103.c     
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief determine the length of string.
 * @version 1.0
 * @date 2022-06-16
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int len(char *ptr)
{
    int count = 0, i = 0;
    while (*(ptr + i) != '\0')
    {
        count++;
        i++;
    }
    return count;
}
int main()
{
    char str[20], *ptrstr;

    printf("Enter the string: ");
    gets(str);
    ptrstr = &str[0];

    printf("The length of string is :%d\n", len(ptrstr));
    return 0;
}


/**
 * @file s101.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief display a value of different data type and its address.
 * @version 1.0
 * @date 2022-06-16
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{
    int intvar, *intptr;
    float floatvar, *floatptr;
    char charvar, *charptr;

    printf("Enter the char :\n");
    scanf("%c", &charvar);
    printf("Enter the integer number :\n");
    scanf("%d", &intvar);
    printf("Enter the float number :\n");
    scanf("%f", &floatvar);

    intptr = &intvar;
    floatptr = &floatvar;
    charptr = &charvar;

    printf("the integer number is:%d and its address is :%u\n", *intptr, intptr);
    printf("the integer number is:%f and its address is :%u\n", *floatptr, floatptr);
    printf("the integer number is:%c and its address is :%u\n", *charptr, charptr);

    return 0;
}
/**
 * @file s1e2.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Changing two numbers without using a third parameter.
 * @version 1.0
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include <stdio.h>

int main(){

    int a,b;

    printf("Enter the First number: ");
    scanf("%d",&a);
    printf("Enter the Second number: ");
    scanf("%d",&b);

    a = a+b;
    b = a-b;
    a = a-b;

    printf("The exchange is:\n a = %d\n b = %d\n",a,b);

    return 0;
}
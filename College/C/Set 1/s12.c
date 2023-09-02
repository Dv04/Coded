/**
 * @file s12.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Calculator of two numbers.
 * @version 1.0
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include <stdio.h>

int main(){

    int a, b;

    printf("Enter 1st Number : ");
    scanf("%d",&a);
    printf("Enter 2nd Number : ");
    scanf("%d",&b);

    float div = (float)a/b;

    printf("sum: %d\n",a+b);
    printf("difference: %d\n",a-b);
    printf("multiplication: %d\n",a*b);
    printf("division: %.3f\n",div);
    printf("Mod: %d\n",a%b); 
    printf("\n");

    return 0;
}
/**
 * @file s1e1.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief 5 Numbers seperated by commas.
 * @version 1.0
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include <stdio.h>

int main(){

    int a,b,c,d,e;

    printf("Enter fist number: ");
    scanf("%d",&a);
    printf("Enter second number: ");
    scanf("%d",&b);
    printf("Enter third number: ");
    scanf("%d",&c);
    printf("Enter fourth number: ");
    scanf("%d",&d);
    printf("Enter fifth number: ");
    scanf("%d",&e);

    printf("%d,%d,%d,%d,%d\n",a,b,c,d,e);

    return 0;
}
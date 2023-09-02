/**
 * @file s1e3.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Sum and Average of 5 given numbers.
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

    int sum, average;

    sum = a + b + c + d + e;
    average = sum /5;
    printf("The sum and average of numbers are:\n Sum = %d\n Average = %d\n",sum,average);

    return 0;
}
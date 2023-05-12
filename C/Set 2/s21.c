/**
 * @file s21.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief To find simple interest.
 * @version 1.0
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include <stdio.h>

int main(){

    int p,r,n;

    printf("Enter Principal Amount: ");
    scanf("%d",&p);
    printf("Enter the Rate: ");
    scanf("%d",&r);
    printf("Enter the number of years: ");
    scanf("%d",&n);

    int SI = (p*r*n)/100;

    printf("The Simple Interest of given data is : %d\n", SI);

    return 0;
}
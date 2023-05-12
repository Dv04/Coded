/**
 * @file s4e1.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Divisible by 5
 * @version 1.0
 * @date 2022-04-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int main(){

    int num;

    printf("Enter the number: ");
    scanf("%d",&num);

    if (num%5==0)
    {
        printf("%d is divisible by 5\n",num);
    }
    else
    {
        printf("Not divisible by 5\n");
    }
    

    return 0;
}
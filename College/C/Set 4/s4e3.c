/**
 * @file s4e1.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Checking the sign
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

    if (num>0)
    {
        printf("%d is positive\n",num);
    }
    else if (num<0 )
    {
        printf("%d is negative\n",num);
    }
    else
    {
        printf("Number is 0\n");
    }
    

    return 0;
}
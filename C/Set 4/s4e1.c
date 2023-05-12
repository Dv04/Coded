/**
 * @file s4e1.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Leap year
 * @version 1.0
 * @date 2022-04-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int main(){

    int year;

    printf("Enter the year: ");
    scanf("%d",&year);

    if (year%4==0)
    {
        printf("Leap year: %d\n",year);
    }
    else
    {
        printf("Not leap year\n");
    }
    

    return 0;
}
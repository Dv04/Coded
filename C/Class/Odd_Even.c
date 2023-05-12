/**
 * @file Odd_Even.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Odd or even
 * @version 1.0
 * @date 2022-04-27
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int main(){

    float x;
    printf("Enter the number: ");
    scanf("%f",&x);

    if((int)x%2==0)
        printf("Even\n");
    else 
        printf("Odd\n");


    return 0;
}
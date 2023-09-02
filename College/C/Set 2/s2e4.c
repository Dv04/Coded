/**
 * @file s2e4.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief To add individual digit of 5 nos.
 * @version 1.0
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int main(){

    int num;
    int r=0;

    printf("Enter a five digit number: ");
    scanf("%d",&num);

    // int count = 0;

    while (num != 0){
        r += num%10;
        num = num/10;
        // count = count + 1;
        // if(count>5){
        //     break;
        // }
    }

    printf("The final sum is : %d\n",r);
    return 0;
}
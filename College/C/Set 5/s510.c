/**
 * @file s510.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Sum of 10 with skip at negative
 * @version 1.0
 * @date 2022-05-19
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main(){

    int sum = 0,num;
    for (int i = 0; i < 10; i++)
    {
        printf("enter a number to add: ");
        scanf("%d", &num);
        if(num > 0){
            sum += num;
        }
    }
    printf("%d\n",sum);

    return 0;
}
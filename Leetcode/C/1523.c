/**
 * @file 1523.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Count odd numbers in even range
 * @version 1.0
 * @date 2023-02-13
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <stdio.h>

int Ans(int x, int y){
    int count = 0;
    if(x-y+1 % 2 == 0){
        count = (x-y+1)/2;
    }
    else{
        count = (x-y+1)/2 + 1;
    }
    printf("%d", count);
    
}

int main(){

    int h, l;
    scanf("%d %d", &l, &h);
    Ans(h, l);

    return 0;
}
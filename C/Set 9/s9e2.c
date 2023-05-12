/**
 * @file s9e2.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Maximum of 3 numbers function
 * @version 1.0
 * @date 2022-04-23 
 * @copyright Copyright (c) 2022
 */

#include <stdio.h>

int max(int a, int b, int c){
    if (a<b)
    {
        if (b<c) printf("Number 3 is the largest number\n");
        else printf("Number 2 is the largest number\n");
    }
    else
    {
        if (a<c) printf("Number 3 is the largest number\n");  
        else printf("Number 1 is the largest number\n"); 
    }
    return 0;
}

int main(){
    int a,b,c;
    printf("Number 1: "); scanf("%d",&a);
    printf("Number 2: "); scanf("%d",&b);
    printf("Number 3: "); scanf("%d",&c);
    max(a,b,c);
    return 0;
}
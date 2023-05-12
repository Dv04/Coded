/**
 * @file s91.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief First N numbers
 * @version 1.0
 * @date 2022-05-31
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int Add(int n){
    int sum = 0;
    for(int i=1; i<=n; i++){
        sum += i;
    }
    printf("%d\n",sum);
    return 0;
}

int main(){

    int n;
    printf("Enter number of numbers to add: ");
    scanf("%d",&n);
    Add(n);

    return 0;
}
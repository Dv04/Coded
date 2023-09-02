/**
 * @file s92.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Factorial using recursion
 * @version 1.0
 * @date 2022-05-31
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>
int Fact(int n){
    int fact = 1;
    while(n>0){
        fact *= n;
        n--;
    }
    printf("fact = %d\n", fact);
    return 0;
}
int main(){

    int n;
    printf("Enter number of numbers to facoriate: ");
    scanf("%d",&n);
    Fact(n);

    return 0;
}
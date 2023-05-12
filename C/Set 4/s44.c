/**
 * @file s44.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Vowel or Constant
 * @version 1.0
 * @date 2022-04-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>
#include <ctype.h>
int main(){

char c;

    printf("Enter a single character: ");
    scanf("%c",&c);
    c = tolower(c);
    switch(c){
        case 'a':
        case 'e':
        case 'i':
        case 'o':
        case 'u':
            printf("It is a vowel.\n"); break;
        
        default: printf("It is a consonant.\n"); break;
    }

    return 0;
}
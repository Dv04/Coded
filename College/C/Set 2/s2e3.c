/**
 * @file s2e3.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Changing Celsius to Fahrenheit temperature and vice versa.
 * @version 1.0
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <stdio.h>

int main(){

    float celcius,fahrenheit;
    char temperature;
    printf("Enter C to enter value in celcius and F to enter value in Fahrenheit: ");
    scanf("%c",&temperature);
    switch(temperature){
        case 'C':
        case 'c':
            printf("\nEnter the value of temperature: ");
            scanf("%f",&celcius);
            fahrenheit = (1.8*celcius)+32;
            printf("The value of temperature in fahrenheit is %.3f\n",fahrenheit);
            break;
        case 'F':
        case 'f':
            printf("\nEnter the value of temperature: ");
            scanf("%f",&fahrenheit);
            celcius = (fahrenheit-32)/1.8;
            printf("The value of temperature in celcius is: %.3f\n",celcius);
            break;
        default: printf("Please enter valid input of character\n"); break;
    }
    return 0;
}
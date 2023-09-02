/**
 * @file s32.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Bitwise operator.
 * @version 1.0
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int main()
{

    unsigned int a , b;
    int c;

    printf("Enter the first number: ");
    scanf("%d", &a);
    
    printf("Enter the second number: ");
    scanf("%d", &b);

    c = a & b;
    printf("Line 1 - Value of c is %d\n", c );

    c = a | b;
    printf("Line 2 - Value of c is %d\n", c );

    c = a ^ b;
    printf("Line 3 - Value of c is %d\n", c );

    c = ~a;
    printf("Line 4 - Value of c is %d\n", c );

    c = a << 2;
    printf("Line 5 - Value of c is %d\n", c );

    c = a >> 2;
    printf("Line 6 - Value of c is %d\n", c );
  
    return 0;
    
}
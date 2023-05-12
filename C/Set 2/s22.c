/**
 * @file s22.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Find roots of equation.
 * @version 1.0
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include <stdio.h>
#include <math.h>

int main(){

    int a = 3,b = 8,c = 2;

    float d = (b*b)-(4*a*c);

    float root1,root2;
    root1 = ((-b)+(sqrt(d)))/(2*a);
    root2 = ((-b)-(sqrt(d)))/(2*a);
    printf("root1 = %f, root2 = %f\n", root1, root2);
    return 0;
}
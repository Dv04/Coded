/**
 * @file Operators.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Operators
 * @version 1.0
 * @date 2022-04-25
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

int main()
{

    int x = 4;
    float y = 3.5;
    char a = 'a', A = 'A';

    printf("%d\n", abs(x));
    printf("%.3f\n", fabsf(y));
    printf("%.3f\n", ceil(x));
    printf("%.3f\n", exp(x));
    printf("%c\n", tolower(A));
    printf("%c\n", toupper(a));
    printf("%.3f\n", sqrt(x));
    printf("%d\n", rand());

    size_t n;
    int buf = sizeof(n);

    switch (buf)
    {
    case 1:
        printf("This is a Character\n");
        break;

    case 4:
        printf("This is an Integer or Floating point\n");
        break;

    case 8:
        printf("This is an (long) long int or (long) double or size_t\n");
        break;

    default:
        printf("This is anotehr type of data");
    }


    return 0;
}
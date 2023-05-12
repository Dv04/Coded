/**
 * @file s43.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Switch statement
 * @version 1.0
 * @date 2022-04-22
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <math.h>

int main()
{

    int a, b, c;
start:
    printf("\n\nEnter the first number: ");
    scanf("%d", &a);
    printf("Enter the second number: ");
    scanf("%d", &b);

    printf("\n\nWhat do you want to do:\n 1 for Addition\n 2 for Subtraction\n 3 for multiplication\n 4 for division\n 5 for mod\n 6 for power raising\n Enter your choice: ");
    scanf("%d", &c);
    int x = a % b, y = pow(a, b);
    printf(" ");
    switch (c)
    {
    case 1:
        printf("%d\n", a + b);
        break;
    case 2:
        printf("%d\n", a - b);
        break;
    case 3:
        printf("%d\n", a * b);
        break;
    case 4:
        printf("%.3f\n", (float)a / b);
        break;
    case 5:
        printf("%d\n", x);
        break;
    case 6:
        printf("%d\n", y);
        break;
    default:
        printf("\n\tError\nEnter a valid number.\n\n");
        break;
    }
    if (a < 0 || b < 0)
    {
        printf("A negative number.\nLoop Exited.\n\n");
        goto exit;
    }
    goto start;
exit:
    return 0;
}
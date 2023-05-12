/**
 * @file s5e2.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Print a series 1 + 1/2 + 1/3.........
 * @version 1.0
 * @date 2022-05-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int num;
    float sum = 1;
    printf("Enter a number: ");
    scanf("%d", &num);

    int temp = num + 2;
    printf("\nThe series is:\n1 + ");
    while (num > 1)
    {
        sum += 1 / ((float)(temp - num));
        printf("1/%d", temp - num);
        if (num != 2)
        {
            printf(" + ");
        }
        num--;
    }

    printf("\nThe sum is: %.3f\n", sum);
    return 0;
}
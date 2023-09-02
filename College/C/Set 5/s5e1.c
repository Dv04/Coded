/**
 * @file s5e1.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Multiplication table of given no
 * @version 1.0
 * @date 2022-05-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int count = 1, num, i = 1;
    printf("Enter a number: ");
    scanf("%d", &num);
    while (count <= 10)
    {
        printf("\n%d x %d = %d\n", num, count, count * num);
        count++;
    }
    return 0;
}
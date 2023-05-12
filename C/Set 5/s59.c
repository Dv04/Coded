/**
 * @file s59.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Stop at -1
 * @version 1.0
 * @date 2022-05-19
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    static int num[100], i;
    do
    {
        i++;
        printf("Enter a number: ");
        scanf("%d", &num[i - 1]);
    } while (num[i - 1] != -1);

    i = 1;
    printf("\n");
    while (num[i] != 0)
    {
        ++i;
        printf("%d\n", num[i - 2]);
    }
    return 0;
}
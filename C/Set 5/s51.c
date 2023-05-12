/**
 * @file s51.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief leap years between years
 * @version 1.0
 * @date 2022-04-28
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int start, end, count = 0;

    printf("Starting year: ");
    scanf("%d", &start);
    printf("Ending year: ");
    scanf("%d", &end);

    while (start <= end)
    {
        if (start % 4 == 0)
        {
            printf("%d\n", start);
            count++;
        }
        start++;
    }

    printf("Number of years: %d\n", count);

    return 0;
}
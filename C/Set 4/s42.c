/**
 * @file s42.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Number of roots
 * @version 1.0
 * @date 2022-07-10
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    if (b * b - 4 * a * c == 0)
    {
        printf("There are two solutions, both equal in value\n");
    }
    else if (b * b - 4 * a * c > 0)
    {
        printf("There are two solutions, both Real and Different\n");
    }
    else if (b * b - 4 * a * c < 0)
    {
        printf("There are two solutions, both Imaginary\n");
    }

    return 0;
}
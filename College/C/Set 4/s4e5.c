/**
 * @file s4e5.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Percentage
 * @version 1.0
 * @date 2022-05-19
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    float grade;
    printf("grade: ");
    scanf("%f", &grade);
    int gr = ((int)grade) / 10;
    switch (gr)
    {
    case 10:
    case 9:
        printf("A+");
        break;
    case 8:
        printf("A");
        break;
    case 7:
        printf("B+");
        break;
    case 6:
        printf("B");
        break;
    case 5:
        printf("C+");
        break;
    case 4:
        printf("C");
        break;
    default:
        printf("F");
        break;
    }
    printf("\n");
    return 0;
}
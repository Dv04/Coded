/**
 * @file project.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Marks
 * @version 1.0
 * @date 2022-05-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int M = 67, C = 67, P = 86, Y = 99, Co = 78;
    float result1, result2, result3, Avg;

    Avg = (float)(M + P + Y + C + Co) / 5;
    result1 = (float)(M + P) / 2;
    result2 = (float)(M + Y) / 2;
    result3 = (float)(M + C) / 2;

    printf("Average: %.3f\n", Avg);

    if (result1 > (75 * Avg / 100))
        printf("With %.3f percentage, you get A+ grade\n", result1);
    else if (result2 > (95 * Avg / 100))
        printf("With %.3f percentage, you get A- grade\n", result2);
    else if (result3 > (70 * Avg / 100))
        printf("With %.3f percentage, you get B+ grade\n", result3);
    else
        printf("with %.3f percentage, you get B- grade\n", Avg);
    return 0;
}
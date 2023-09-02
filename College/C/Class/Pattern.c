/**
 * @file Pattern.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Patterns
 * @version 1.0
 * @date 2022-05-17
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <ctype.h>
int main()
{

    // int i, j, n = 4;
    // for (i = 1; i <= n; i++)
    // {
    //     for (j = n; j > i; j--)
    //     {
    //         printf("  ");
    //     }
    //     for (j = 0; j < i; j++)
    //     {
    //         printf(" %d", j + 1);
    //     }
    //     for (j = 1; j < i; j++)
    //     {
    //         printf(" %d", j);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    int year;
    char out[12][3] = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"};
    int day[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}, check = 0, isor = 0;
    int i = 1;
    char in[10];
    char input[10];

    printf("Enter the Month of choice: ");
    scanf("%s", in);
    while (i < 4)
    {
        input[i - 1] = toupper(in[i - 1]);
        i++;
    }
    while (check < 12)
    {
        for (int i = 0; i < 3; i++)
        {
            if (input[i] == out[check][i])
            {
                isor = 1;
            }
            else
            {
                isor = 0;
                break;
            }
        }
        if (isor == 1)
        {
            if (check == 1)
            {
                printf("Enter year: ");
                scanf("%d", &year);
                if (year % 4 == 0)
                {
                    printf("The days are 29\n");
                    break;
                }
                else
                {
                    printf("The days are 28\n");
                    break;
                }
            }
            else
            {
                printf("The days are: %d\n", day[check]);
                break;
            }
        }
        check++;
    }
    if (isor == 0)
    {
        printf("\tERROR ERROR\nYou entered incorrect month\n");
    }
    return 0;
}
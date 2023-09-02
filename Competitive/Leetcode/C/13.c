/**
 * @file 13.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief 13th program, Given a roman numeral, convert it to an integer.
 * @version 1.0
 * @date 2022-11-28
 *
 * @copyright Copyright (c) 2022
 *
 */

// Given a roman numeral, convert it to an integer.

#include <stdio.h>
#include <string.h>
#include <ctype.h>

int romanToInt(char *s)
{
    int i, sum = 0;
    for (i = 0; i < strlen(s); i++)
    {
        if (s[i] == 'M')
        {
            sum += 1000;
        }
        else if (s[i] == 'D')
        {
            sum += 500;
        }
        else if (s[i] == 'C')
        {
            if (s[i + 1] == 'D')
            {
                sum += 400;
                i++;
            }
            else if (s[i + 1] == 'M')
            {
                sum += 900;
                i++;
            }
            else
            {
                sum += 100;
            }
        }
        else if (s[i] == 'L')
        {
            sum += 50;
        }
        else if (s[i] == 'X')
        {
            if (s[i + 1] == 'L')
            {
                sum += 40;
                i++;
            }
            else if (s[i + 1] == 'C')
            {
                sum += 90;
                i++;
            }
            else
            {
                sum += 10;
            }
        }
        else if (s[i] == 'V')
        {
            sum += 5;
        }
        else if (s[i] == 'I')
        {
            if (s[i + 1] == 'V')
            {
                sum += 4;
                i++;
            }
            else if (s[i + 1] == 'X')
            {
                sum += 9;
                i++;
            }
            else
            {
                sum += 1;
            }
        }
    }
    return sum;
}

int main()
{
    char s[100];
    printf("Enter the roman number: ");
    scanf("%s", s);
    char S = toupper(s);
    printf("The integer value is: %d\n", romanToInt(S));
    return 0;
}
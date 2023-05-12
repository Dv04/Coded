/**
 * @file Function.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief A function in c is a specific code snippet designed to execute and perform user defined task.
 * @brief Function that returns index of first uppercase character of a string
 * @version 1.0
 * @date 2022-05-30
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>

// int str(char s)
// {
//     if (isupper(s))
//     {
//         return 1;
//     }
//     else
//     {
//         return 0;
//     }
// }
// int main()
// {
//     char s[100];
//     printf("Enter your string: ");
//     scanf("%s", &s);
//     for (int i = 0; i < strlen(s); i++)
//     {
//         int a = str(s[i]);
//         if (a == 1)
//         {
//             printf("%d\n", i+1);
//             break;
//         }
//     }
//     return 0;
// }
//// HW same function name with different return type
// struct Test{
//     char name[100];
//     int marks;
// };

// int name(char a[], char b[])
// {
//     int i, j = 0, k = 0;
//     for (i = 0; i < strlen(a) || i < strlen(b); i++)
//     {
//         if (a[i] == b[i])
//         {
//             j++;
//             // printf("1");
//         }
//         else
//         {
//             k++;
//             // printf("0");
//         }
//     }
//     return printf("%d\n", abs(j - k));
// }
// int main()
// {
//     char a[100], b[100];
//     printf("Enter the string one: ");
//     scanf("%s", a);
//     printf("Enter the string two: ");
//     scanf("%s", b);
//     name(a, b);
//     return 0;
// }

// when u pass a value of variable directly is known as pass by value

// write a function that takes height and weight of 5 students and calculates BMI

// float BMI(int h, int w)
// {
//     float bmi;
//     bmi = w / pow(h, 2);
//     printf("%f\n\n", bmi);
//     return bmi;
// }
// float avg(float av[5])
// {
//     float sum = 0;
//     for (int i = 0; i < 5; i++)
//     {
//         sum += av[i];
//     }
//     return sum / 5;
// }
// int main()
// {
//     float h, w, av[5], ans;
//     for (int i = 0; i < 5; i++)
//     {
//         printf("Enter height of person %d: ", i + 1);
//         scanf("%f", &h);
//         printf("Enter weight of person %d: ", i + 1);
//         scanf("%f", &w);
//         av[i] = BMI(h, w);
//     }
//     printf("\nThe average BMI of given 5 person is: %.3f\n", avg(av));
//     return 0;
// }

// float grade(float mark[])
// {
//     float avg = 0, sd = 0;
//     for (int i = 0; i < 10; i++)
//     {
//         avg += mark[i];
//     }
//     avg /= 10;

//     for (int i = 0; i < 10; i++)
//     {
//         sd += pow(mark[i] - (avg), 2);
//     }
//     sd = sqrt(sd / 10);
//     // printf("%.3f %.3f\n", avg, sd);
//     return avg;
//     return sd;
//     // return 0;
// }

// int main()
// {
//     float mark[10];
//     float mean, sd;
//     printf("Enter marks: \n");
//     for (int i = 0; i < 10; i++)
//     {
//         scanf("%f", &mark[i]);
//     }
//     // grade(mark);
//     // mean = grade(mark);
//     printf("%.3f %.3f\n", grade(mark), grade(mark));
//     return 0;
// }

int fact(long int n, char s[100])
{
    int count = 1;
    for (int i = 0; i < strlen(s); i++)
    {
        if (islower(s[i]))
        {
            printf("%c\n", s[i]);
            n++;
        }
    }
    printf("%ld\n", n);

    for (int i = 1; i <= n; i++)
    {
        count *= i;
        printf("%d\n", count);
    }
    return count;
}
int main()
{
    char de[100] = "HoWorld";
    long double i = 2.4;
    printf("%.3lf %.3lf\n", i, pow(10.2, 1 / i));
    long int n = fact(0, de);
}
// a function that calls itself is called a recursive function and this technique is known as recursion.

// we use double when we deal with exponents.
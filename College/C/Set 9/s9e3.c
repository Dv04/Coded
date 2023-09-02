/**
 * @file s9e3.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief 1 if prime and 0 if not prime
 * @version 1.0
 * @date 2022-05-31
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
int Prime(int n)
{
    int count = 1;
    for (int i = 1; i <= (n / 2); i++)
    {
        if (n % i == 0)
        {
            count++;
        }
    }
    if (count > 2)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}
int main()
{

    int num;
    printf("Enter number: ");
    scanf("%d", &num);
    int bo = Prime(num);
    printf("%d\n", bo);

    return 0;
}
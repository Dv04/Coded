/**
 * @file Recursion.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Recursive functions
 * @version 1.0
 * @date 2022-06-01
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int A(int m, int n)
{
    if (m == 0)
    {
        return n + 1;
    }
    else if (m > 0 && n == 0)
    {
        return A(m - 1, 1);
    }
    else if (m > 0 && n > 0)
    {
        return A(m - 1, A(m, n - 1));
    }

    return 0;
}
int main()
{

    int m, n;
    printf("Enter 2 number: ");
    scanf("%d %d", &m, &n);
    printf("%d\n", A(m, n));
    return 0;
}

/** n+1 if m = 0
 *  A(m-1,1) if m>0, n = 0
 *  A(m-1,A(m,n-1)) if m>0, n>0
 */
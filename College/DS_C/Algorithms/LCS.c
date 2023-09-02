/**
 * @file LCS.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Longest Common Sequence
 * @version 1.0
 * @date 2023-01-18
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX 100

int max(int a, int b)
{
    return (a > b) ? a : b;
}

int longestCommonSubsequence(char *X, char *Y)
{
    int m = strlen(X);
    int n = strlen(Y);
    int L[m + 1][n + 1];
    int i, j;

    for (i = 0; i <= m; i++)
    {
        for (j = 0; j <= n; j++)
        {
            if (i == 0 || j == 0)
            {
                L[i][j] = 0;
            }
            else if (X[i - 1] == Y[j - 1])
            {
                L[i][j] = L[i - 1][j - 1] + 1;
            }
            else
            {
                L[i][j] = max(L[i - 1][j], L[i][j - 1]);
            }
        }
    }
    return L[m][n];
}

int main()
{
    char X[MAX], Y[MAX];
    printf("Enter the first string: ");
    scanf("%s", X);
    printf("Enter the second string: ");
    scanf("%s", Y);

    printf("Length of LCS is %d", longestCommonSubsequence(X, Y));
    printf("\n");
    return 0;
}
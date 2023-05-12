/**
 * @file s62.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Multiply matrix
 * @version 1.0
 * @date 2022-05-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>

int main()
{

    int n, r1, r, c, c2, i, j;
    printf("\n\nEnter matrix 1 size: ");
    scanf("%d %d", &r1, &c);
    printf("Enter matrix 2 size: ");
    scanf("%d %d", &r, &c2);

    int F[r1][c], S[r][c2];
    int A[r1][c2];
    if (r == c)
    {
        n = r = c;
        printf("\n");
        for (i = 0; i < r1; i++)
        {
            for (j = 0; j < n; j++)
            {
                printf("Element of Matrix 1 at %dx%d: ", i + 1, j + 1);
                scanf("%d", &F[i][j]);
            }
        }
        printf("\n");
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < c2; j++)
            {
                printf("Element of Matrix 2 at %dx%d: ", i + 1, j + 1);
                scanf("%d", &S[i][j]);
            }
        }

        for (i = 0; i < r1; i++)
        {
            for (j = 0; j < c2; j++)
            {
                A[i][j] = 0;
                for (int k = 0; k < n; k++)
                {
                    A[i][j] += F[i][k] * S[k][j];
                }
            }
        }
        printf("\nElement of Answer Matrix: \n\n");
        for (i = 0; i < r1; i++)
        {
            printf("|");
            for (j = 0; j < c2; j++)
            {
                printf("  %d", A[i][j]);
            }

            printf(" |\n");
        }
        printf("\n\n");
    }
    else
    {
        printf("\n\t\tError\nThe required Rows and Columns not matching\n\n");
    }
    return 0;
}
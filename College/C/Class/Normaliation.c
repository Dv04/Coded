#include <stdio.h>
#include <math.h>

int main()
{
    int rows, cols, r, c;

    printf("Enter size of a matrix\n");
    scanf("%d %d", &rows, &cols);

    long long int a, square[cols];
    double inputMatrix[rows][cols];
    double norm[rows][cols];

    printf("Enter matrix of size %dX%d\n", rows, cols);
    for (r = 0; r < rows; r++)
    {
        for (c = 0; c < cols; c++)
        {
            scanf("%lf", &inputMatrix[r][c]);
        }
    }

    for (r = 0; r < cols; r++)
    {
        square[r] = 0;
        for (c = 0; c < rows; c++)
        {
            a = inputMatrix[c][r];
            square[r] += a * a;
            // printf("%lld", a);
        }
        printf("Sum of squares of column %d: %lld\n", r, square[r]);
    }

    for (r = 0; r < cols; r++)
    {
        for (c = 0; c < rows; c++)
        {
            norm[c][r] = inputMatrix[c][r] / sqrt(square[r]);
        }
    }

    printf("\nNormalized Matrix:\n");
    for (r = 0; r < rows; r++)
    {
        for (c = 0; c < cols; c++)
        {
            printf("%.3lf ", norm[r][c]);
        }
        printf("\n");
    }

    return 0;
}
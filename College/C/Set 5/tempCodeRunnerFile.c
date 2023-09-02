#include <stdio.h>

int main()
{

    int i, j, n = 5;
    for (i = n; i > 0; i--)
    {
        for (j = n; j > i; j--)
        {
            printf("  ");
        }
        for (j = 0; j < i; j++)
        {
            printf(" %d", j + 1);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}
#include <stdio.h>

int main()
{

    int n;
    printf("How many number: ");
    scanf("%d", &n);

    int arr[n];
    for (int i = 1; i <= n; i++)
    {
        printf("Enter a number: ");
        scanf("%d", &arr[n - i]);
    }
    printf("\n");
    for (int i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
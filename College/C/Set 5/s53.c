#include <stdio.h>

int main()
{
    int n, temp1, ans = 0;
    printf("How many numbers do you want to input: ");
    scanf("%d", &n);
    while (n > 0)
    {
        printf("Enter a number: ");
        scanf("%d", &temp1);
        if (temp1 > ans)
        {
            ans = temp1;
        }
        n--;
    }
    printf("Answer is: %d\n", ans);
    return 0;
}